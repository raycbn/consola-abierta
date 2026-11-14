from __future__ import annotations
import csv, json, math, os, platform, socket, time
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE, REPORTS, VISUALS = ROOT/"evidence", ROOT/"reports", ROOT/"evidence"/"visuals"


def attention(q, k, v, causal=True):
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if causal:
        tq, tk = scores.shape[-2:]
        mask = torch.triu(torch.ones(tq, tk, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    p = torch.softmax(scores, dim=-1)
    return p @ v, p


def manual_gradients(q, k, v, p, out, causal=True):
    # Exact reverse-mode derivatives for L = mean(out ** 2).
    do = 2.0 * out / out.numel()
    dv = p.transpose(-2, -1) @ do
    dp = do @ v.transpose(-2, -1)
    ds = p * (dp - (dp * p).sum(dim=-1, keepdim=True))
    if causal:
        tq, tk = ds.shape[-2:]
        mask = torch.triu(torch.ones(tq, tk, dtype=torch.bool), diagonal=1)
        ds = ds.masked_fill(mask, 0.0)
    scale = 1.0 / math.sqrt(q.shape[-1])
    dq = (ds @ k) * scale
    dk = (ds.transpose(-2, -1) @ q) * scale
    return dq, dk, dv


def scaling_experiment():
    gen = torch.Generator().manual_seed(20261115)
    rows = []
    for d in (8, 16, 32, 64, 128, 256):
        q = torch.randn(512, 1, d, generator=gen)
        k = torch.randn(512, 24, d, generator=gen)
        raw = (q @ k.transpose(-2, -1)).squeeze(1)
        for mode, scores in (("unscaled", raw), ("scaled", raw / math.sqrt(d))):
            p = torch.softmax(scores, dim=-1)
            entropy = (-(p * p.clamp_min(1e-12).log()).sum(-1)).mean()
            rows.append({
                "d_head": d, "mode": mode,
                "mean_entropy_nats": float(entropy),
                "mean_max_attention_weight": float(p.max(-1).values.mean()),
                "samples": 512, "keys_per_query": 24
            })
    return rows


def gradient_experiment():
    torch.manual_seed(20261115)
    dtype = torch.float64
    b, t, dm, h = 2, 7, 32, 4
    dh = dm // h
    x = torch.randn(b, t, dm, dtype=dtype, requires_grad=True)
    target = torch.randn(b, t, dm, dtype=dtype)
    ws = [(torch.randn(dm, dm, dtype=dtype) / math.sqrt(dm)).requires_grad_() for _ in range(4)]
    wq, wk, wv, wo = ws

    def forward():
        q = (x @ wq).view(b, t, h, dh).transpose(1, 2)
        k = (x @ wk).view(b, t, h, dh).transpose(1, 2)
        v = (x @ wv).view(b, t, h, dh).transpose(1, 2)
        out, p = attention(q, k, v)
        merged = out.transpose(1, 2).contiguous().view(b, t, dm)
        return (merged @ wo - target).square().mean(), q, k, v

    loss, q, k, v = forward()
    gx, gq, gk, gv, go = torch.autograd.grad(loss, (x, wq, wk, wv, wo))
    eps = 1e-6
    original = float(wq[0, 0].detach())
    with torch.no_grad(): wq[0, 0] = original + eps
    lp = float(forward()[0].detach())
    with torch.no_grad(): wq[0, 0] = original - eps
    lm = float(forward()[0].detach())
    with torch.no_grad(): wq[0, 0] = original
    finite = (lp - lm) / (2 * eps)

    ql, kl, vl = (z.detach().clone().requires_grad_(True) for z in (q, k, v))
    out, p = attention(ql, kl, vl)
    aq, ak, av = torch.autograd.grad(out.square().mean(), (ql, kl, vl))
    mq, mk, mv = manual_gradients(ql.detach(), kl.detach(), vl.detach(), p.detach(), out.detach())
    err = max(float((a-m).abs().max()) for a,m in ((aq,mq),(ak,mk),(av,mv)))
    return {
        "dtype": "float64", "batch": b, "sequence_length": t, "d_model": dm,
        "query_heads": h, "head_dimension": dh, "loss": float(loss.detach()),
        "gradient_l2_norms": {
            "input_x": float(gx.norm()), "W_Q": float(gq.norm()), "W_K": float(gk.norm()),
            "W_V": float(gv.norm()), "W_O": float(go.norm())
        },
        "finite_difference": {
            "parameter": "W_Q[0,0]", "epsilon": eps,
            "autograd": float(gq[0,0]), "central_difference": finite,
            "absolute_error": abs(float(gq[0,0])-finite)
        },
        "manual_vs_autograd_gradient_max_abs_error": err
    }


def rope(x, position):
    d = x.shape[-1]
    inv = 1.0 / (10000.0 ** (torch.arange(0, d, 2, dtype=x.dtype) / d))
    angle = position * inv
    c, s = angle.cos(), angle.sin()
    out = torch.empty_like(x)
    even, odd = x[..., 0::2], x[..., 1::2]
    out[..., 0::2] = even*c - odd*s
    out[..., 1::2] = even*s + odd*c
    return out


def rope_check():
    torch.manual_seed(20261115)
    q, k = torch.randn(32, dtype=torch.float64), torch.randn(32, dtype=torch.float64)
    m, n = 5, 19
    return float((torch.dot(rope(q,m), rope(k,n)) - torch.dot(q,rope(k,n-m))).abs())


def gqa_sdpa_check():
    torch.manual_seed(20261115)
    b, hq, hkv, t, dh = 2, 8, 2, 12, 8
    q = torch.randn(b, hq, t, dh)
    k = torch.randn(b, hkv, t, dh)
    v = torch.randn(b, hkv, t, dh)
    copies = hq // hkv
    kr, vr = k.repeat_interleave(copies, dim=1), v.repeat_interleave(copies, dim=1)
    manual, p = attention(q, kr, vr)
    sdpa = F.scaled_dot_product_attention(q, kr, vr, dropout_p=0.0, is_causal=True)
    future = torch.triu(torch.ones(t,t,dtype=torch.bool), diagonal=1).view(1,1,t,t)
    return {
        "query_shape": list(q.shape), "key_value_shape": list(k.shape),
        "expanded_key_value_shape": list(kr.shape), "output_shape": list(manual.shape),
        "group_size": copies,
        "manual_vs_pytorch_sdpa_max_abs_delta": float((manual-sdpa).abs().max()),
        "causal_future_mass_max": float(p.masked_select(future).abs().max()),
        "attention_row_sum_max_abs_error": float((p.sum(-1)-1).abs().max())
    }


def variant_tables():
    projections, cache = [], []
    dm, hq, dh = 512, 8, 64
    for hkv in (8,4,2,1):
        qn, kvn, on = dm*hq*dh, 2*dm*hkv*dh, hq*dh*dm
        total = qn + kvn + on
        projections.append({
            "d_model":dm, "query_heads":hq, "kv_heads":hkv,
            "attention_projection_parameters_no_bias":total,
            "relative_to_mha_percent":round(total/(4*dm*dm)*100,2)
        })
    # One layer, one sequence, BF16, 4096 cached tokens, 32 Q heads x 128 dims.
    for hkv in (32,8,4,1):
        per_token = 2*hkv*128*2
        cache.append({
            "query_heads":32, "kv_heads":hkv, "head_dimension":128,
            "sequence_tokens":4096, "dtype":"BF16",
            "bytes_per_token_per_layer":per_token,
            "mib_per_layer_per_sequence":round(per_token*4096/(1024**2),3),
            "relative_to_mha_percent":round(hkv/32*100,2)
        })
    return projections, cache


def make_scaling_svg(rows, path):
    dims = sorted({r["d_head"] for r in rows})
    values = {m:[next(x["mean_entropy_nats"] for x in rows if x["d_head"]==d and x["mode"]==m) for d in dims] for m in ("unscaled","scaled")}
    w,ht,left,right,top,bottom = 900,470,90,45,55,90
    pw,ph = w-left-right,ht-top-bottom
    ymax = max(sum(values.values(), []))*1.08
    px=lambda i:left+i*pw/(len(dims)-1)
    py=lambda v:ht-bottom-v/ymax*ph
    s=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 470" role="img" aria-labelledby="t d">',
       '<title id="t">Softmax entropy by attention head dimension</title>',
       '<desc id="d">Synthetic fixed-seed CPU experiment comparing scaled and unscaled dot products for 24 keys.</desc>',
       '<rect width="100%" height="100%" fill="#08111f"/>',
       '<text x="450" y="32" fill="#f0f6ff" text-anchor="middle" font-size="21" font-weight="700">Softmax entropy vs. head dimension</text>']
    for i in range(5):
        v=ymax*i/4; y=py(v)
        s += [f'<line x1="{left}" y1="{y:.1f}" x2="{w-right}" y2="{y:.1f}" stroke="#24394f"/>',
              f'<text x="{left-12}" y="{y+5:.1f}" fill="#9bb0c7" text-anchor="end" font-size="13">{v:.2f}</text>']
    s += [f'<line x1="{left}" y1="{top}" x2="{left}" y2="{ht-bottom}" stroke="#7890a8"/>',
          f'<line x1="{left}" y1="{ht-bottom}" x2="{w-right}" y2="{ht-bottom}" stroke="#7890a8"/>']
    for i,d in enumerate(dims):
        s.append(f'<text x="{px(i):.1f}" y="{ht-bottom+26}" fill="#b8c8d9" text-anchor="middle" font-size="13">{d}</text>')
    for mode,color in (("unscaled","#f59e0b"),("scaled","#22d3ee")):
        pts=" ".join(f"{px(i):.1f},{py(v):.1f}" for i,v in enumerate(values[mode]))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="4"/>')
        for i,v in enumerate(values[mode]):
            s.append(f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="5" fill="{color}"/>')
    s += ['<text x="472" y="443" fill="#b8c8d9" text-anchor="middle" font-size="15">Head dimension (d_k)</text>',
          '<text x="22" y="250" fill="#b8c8d9" text-anchor="middle" font-size="15" transform="rotate(-90 22 250)">Mean entropy (nats)</text>',
          '<line x1="565" y1="65" x2="600" y2="65" stroke="#f59e0b" stroke-width="4"/><text x="610" y="70" fill="#f0f6ff" font-size="14">Unscaled</text>',
          '<line x1="710" y1="65" x2="745" y2="65" stroke="#22d3ee" stroke-width="4"/><text x="755" y="70" fill="#f0f6ff" font-size="14">Scaled by 1/√d_k</text>',
          '<text x="450" y="462" fill="#8499b0" text-anchor="middle" font-size="11">512 random queries · 24 keys/query · CPU · seed 20261115</text></svg>']
    path.write_text("\n".join(s),encoding="utf-8")


def write_visuals():
    flow = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 500" role="img" aria-labelledby="t d"><title id="t">Self-attention forward and backward dataflow</title><desc id="d">Forward tensor flow from X through Q, K and V to scores, softmax and output; backward gradient path through the chain rule.</desc><defs><marker id="a" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8Z" fill="#8aa5bd"/></marker></defs><rect width="100%" height="100%" fill="#08111f"/><text x="540" y="34" fill="#f0f6ff" font-size="24" font-weight="700" text-anchor="middle">Self-Attention · forward pass and gradient flow</text><g font-family="Arial" text-anchor="middle"><rect x="35" y="100" width="100" height="58" rx="9" fill="#102b3a" stroke="#22d3ee"/><text x="85" y="135" fill="#fff" font-size="20">X</text><rect x="195" y="58" width="125" height="55" rx="9" fill="#142d46" stroke="#60a5fa"/><text x="257" y="92" fill="#fff" font-size="18">Q = XWQ</text><rect x="195" y="132" width="125" height="55" rx="9" fill="#142d46" stroke="#60a5fa"/><text x="257" y="166" fill="#fff" font-size="18">K = XWK</text><rect x="195" y="206" width="125" height="55" rx="9" fill="#142d46" stroke="#60a5fa"/><text x="257" y="240" fill="#fff" font-size="18">V = XWV</text><rect x="390" y="92" width="155" height="80" rx="9" fill="#26304a" stroke="#a78bfa"/><text x="467" y="124" fill="#fff" font-size="17">S = QKᵀ / √dₖ</text><text x="467" y="150" fill="#cbbcf5" font-size="14">+ causal mask</text><rect x="620" y="92" width="145" height="80" rx="9" fill="#26304a" stroke="#a78bfa"/><text x="692" y="124" fill="#fff" font-size="17">P = softmax(S)</text><text x="692" y="150" fill="#cbbcf5" font-size="14">row-wise</text><rect x="840" y="92" width="175" height="80" rx="9" fill="#123c37" stroke="#34d399"/><text x="927" y="124" fill="#fff" font-size="20">O = PV</text><text x="927" y="150" fill="#a7f3d0" font-size="14">context vectors</text><g stroke="#8aa5bd" stroke-width="2" fill="none" marker-end="url(#a)"><path d="M135 115 L185 86"/><path d="M135 129 L185 159"/><path d="M135 143 L185 233"/><path d="M320 85 L380 112"/><path d="M320 159 L380 150"/><path d="M545 132 L610 132"/><path d="M765 132 L830 132"/><path d="M320 233 C540 315 745 270 870 185"/></g><text x="540" y="330" fill="#fda4af" font-size="20" font-weight="700">Backward pass · chain rule</text><rect x="840" y="370" width="175" height="70" rx="9" fill="#3b1e2a" stroke="#fb7185"/><text x="927" y="399" fill="#fff" font-size="18">Loss L(O)</text><text x="927" y="423" fill="#fecdd3" font-size="13">upstream gradient</text><rect x="620" y="370" width="145" height="70" rx="9" fill="#302139" stroke="#c084fc"/><text x="692" y="399" fill="#fff" font-size="17">∂L / ∂P</text><text x="692" y="423" fill="#e9d5ff" font-size="13">and ∂L / ∂V</text><rect x="390" y="370" width="155" height="70" rx="9" fill="#302139" stroke="#c084fc"/><text x="467" y="399" fill="#fff" font-size="17">∂L / ∂S</text><text x="467" y="423" fill="#e9d5ff" font-size="13">softmax Jacobian</text><rect x="195" y="370" width="125" height="70" rx="9" fill="#3b2c1b" stroke="#fbbf24"/><text x="257" y="399" fill="#fff" font-size="16">∂L/∂Q,K,V</text><text x="257" y="423" fill="#fde68a" font-size="12">then WQ, WK, WV</text><g stroke="#fb7185" stroke-width="2" fill="none" marker-end="url(#a)"><path d="M840 405 L775 405"/><path d="M620 405 L555 405"/><path d="M390 405 L330 405"/></g><text x="540" y="477" fill="#9bb0c7" font-size="14">Gradients reach Q, K and V through both the weighted sum and the score/softmax path.</text></g></svg>"""
    (VISUALS/"attention-gradient-flow.svg").write_text(flow,encoding="utf-8")
    variants = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1040 390" role="img" aria-labelledby="t d"><title id="t">MHA, GQA and MQA comparison</title><desc id="d">Eight query heads connected to eight, two, or one shared key-value heads.</desc><rect width="100%" height="100%" fill="#08111f"/><text x="520" y="35" text-anchor="middle" fill="#f0f6ff" font-size="24" font-weight="700">Query heads vs. Key/Value heads</text>"""
    for cx,nkv,label in ((170,8,"MHA · 8 KV"),(520,2,"GQA · 2 KV"),(870,1,"MQA · 1 KV")):
        qx=cx-120; kvx=cx+28
        variants+=f'<text x="{cx}" y="78" fill="#e6f1ff" text-anchor="middle" font-size="19" font-weight="700">{label}</text>'
        for i in range(8):
            y=105+i*29
            variants+=f'<rect x="{qx}" y="{y}" width="72" height="21" rx="4" fill="#15314a" stroke="#60a5fa"/><text x="{qx+36}" y="{y+15}" fill="#dbeafe" text-anchor="middle" font-size="11">Q{i+1}</text>'
        for i in range(nkv):
            y=105+i*(232/nkv); hh=min(22,232/nkv-4)
            variants+=f'<rect x="{kvx}" y="{y:.1f}" width="104" height="{hh:.1f}" rx="4" fill="#153d36" stroke="#34d399"/><text x="{kvx+52}" y="{y+15:.1f}" fill="#d1fae5" text-anchor="middle" font-size="11">K{i+1} / V{i+1}</text>'
            for j in range(i*(8//nkv),(i+1)*(8//nkv)):
                qy=115+j*29; ky=y+hh/2
                variants+=f'<path d="M{qx+72} {qy} C{cx-20} {qy} {cx-15} {ky} {kvx} {ky}" stroke="#7890a8" stroke-width="1" fill="none" opacity=".7"/>'
        variants+=f'<text x="{cx}" y="365" fill="#9bb0c7" text-anchor="middle" font-size="14">8 query heads · {nkv} KV heads</text>'
    variants+='</svg>'
    (VISUALS/"attention-variants-mha-gqa-mqa.svg").write_text(variants,encoding="utf-8")


def main():
    for p in (EVIDENCE,REPORTS,VISUALS): p.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    start=time.perf_counter()
    scaling, gradients, gqa = scaling_experiment(), gradient_experiment(), gqa_sdpa_check()
    rope_error=rope_check()
    projection_rows,cache_rows=variant_tables()
    runtime=time.perf_counter()-start
    for name,rows in (("scaling-experiment.csv",scaling),("attention-variants.csv",projection_rows),("kv-cache-estimates.csv",cache_rows)):
        with (EVIDENCE/name).open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    make_scaling_svg(scaling,VISUALS/"attention-scaling-entropy.svg")
    write_visuals()
    checks={
        "manual_gradient_matches_autograd": gradients["manual_vs_autograd_gradient_max_abs_error"]<1e-10,
        "finite_difference_matches_autograd": gradients["finite_difference"]["absolute_error"]<1e-6,
        "causal_mask_blocks_future_positions": gqa["causal_future_mass_max"]==0.0,
        "attention_rows_sum_to_one": gqa["attention_row_sum_max_abs_error"]<1e-6,
        "manual_attention_matches_pytorch_sdpa": gqa["manual_vs_pytorch_sdpa_max_abs_delta"]<1e-5,
        "gqa_kv_expansion_equivalence": gqa["manual_vs_pytorch_sdpa_max_abs_delta"]<1e-5,
        "rope_relative_position_identity": rope_error<1e-10
    }
    metrics={
        "lab":"Consola Abierta - Self-Attention Deep Dive 02",
        "run_timestamp_local":time.strftime("%Y-%m-%d %H:%M:%S"),
        "host":socket.gethostname(),"platform":platform.platform(),
        "python":platform.python_version(),"pytorch":torch.__version__,"device":"cpu",
        "cuda_available":torch.cuda.is_available(),
        "xpu_available":bool(getattr(torch,"xpu",None) and torch.xpu.is_available()),
        "cpu_threads":torch.get_num_threads(),"runtime_seconds":round(runtime,4),
        "scaling_experiment":scaling,"gradient_experiment":gradients,
        "gqa_sdpa_experiment":gqa,"rope_relative_position_max_abs_error":rope_error,
        "attention_projection_estimates":projection_rows,"kv_cache_estimates":cache_rows,
        "checks":checks,
        "limitations":[
            "Synthetic random tensors isolate mathematical properties; they are not model-quality or language-understanding measurements.",
            "The scaling experiment uses a fixed seed and standard-normal Q/K vectors; trained representations may have different statistics.",
            "Parameter and KV-cache tables are analytical estimates for one layer, one sequence and BF16 cache storage, not measured GPU memory.",
            "The CPU lab explicitly expands shared KV heads for GQA; it does not benchmark the native CUDA enable_gqa path.",
            "The RoPE test checks the relative-position rotation identity, not language-model quality or long-context extrapolation.",
            "Attention weights are internal routing coefficients, not a complete or automatically faithful explanation of model decisions."
        ]}
    (REPORTS/"metrics.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding="utf-8")
    report=[
        "# Lab report: Self-Attention deep dive","",f"- Host: {metrics['host']}",f"- OS: {metrics['platform']}",
        f"- Python: {metrics['python']}",f"- PyTorch: {metrics['pytorch']}","- Device: CPU",
        f"- CPU threads: {metrics['cpu_threads']}",f"- Runtime: {runtime:.4f} s","",
        "## Gradient validation","",
        f"- Manual Q/K/V gradients vs. autograd max absolute error: {gradients['manual_vs_autograd_gradient_max_abs_error']:.3e}",
        f"- W_Q[0,0] finite difference vs. autograd absolute error: {gradients['finite_difference']['absolute_error']:.3e}",
        f"- Gradient L2 norms: {json.dumps(gradients['gradient_l2_norms'],ensure_ascii=False)}","",
        "## Scaling experiment","","| d_k | Entropy unscaled | Entropy scaled | Max weight unscaled | Max weight scaled |",
        "|---:|---:|---:|---:|---:|"
    ]
    for d in (8,16,32,64,128,256):
        u=next(r for r in scaling if r["d_head"]==d and r["mode"]=="unscaled")
        s=next(r for r in scaling if r["d_head"]==d and r["mode"]=="scaled")
        report.append(f"| {d} | {u['mean_entropy_nats']:.4f} | {s['mean_entropy_nats']:.4f} | {u['mean_max_attention_weight']:.4f} | {s['mean_max_attention_weight']:.4f} |")
    report += ["","## GQA / SDPA tensor check","",
        f"- Query shape: {gqa['query_shape']}",f"- KV shape before sharing expansion: {gqa['key_value_shape']}",
        f"- Expanded KV shape: {gqa['expanded_key_value_shape']}",f"- Output shape: {gqa['output_shape']}",
        f"- Manual attention vs. PyTorch SDPA max abs delta: {gqa['manual_vs_pytorch_sdpa_max_abs_delta']:.3e}",
        "","## RoPE identity","",f"- Relative-position rotation identity max absolute error: {rope_error:.3e}",
        "","## Automated checks",""]
    report += [f"- {k}: {'PASS' if v else 'FAIL'}" for k,v in checks.items()]
    report += ["","## Limitations",""]+[f"- {x}" for x in metrics["limitations"]]
    (REPORTS/"lab-report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print("=== SELF-ATTENTION LAB SUMMARY ===")
    print(f"host={metrics['host']} python={metrics['python']} torch={metrics['pytorch']} device=cpu")
    print(f"runtime_seconds={runtime:.4f}")
    print(f"gradient_max_abs_error={gradients['manual_vs_autograd_gradient_max_abs_error']:.3e}")
    print(f"finite_difference_error={gradients['finite_difference']['absolute_error']:.3e}")
    print(f"sdpa_max_abs_delta={gqa['manual_vs_pytorch_sdpa_max_abs_delta']:.3e}")
    print(f"rope_identity_error={rope_error:.3e}")
    print("checks="+json.dumps(checks,ensure_ascii=False))
    failed=[k for k,v in checks.items() if not v]
    if failed: print("FAILED: "+", ".join(failed)); return 1
    print("LAB STATUS: PASS"); return 0


if __name__=="__main__":
    raise SystemExit(main())
