#!/usr/bin/env python
"""Inference Engineering lab for Consola Abierta, chapter 04.

CPU-only proof that incremental autoregressive decoding with per-layer K/V
caches matches full-prefix recomputation. This validates mechanics, not model
quality or production serving speed.
"""
from __future__ import annotations

import csv
import json
import math
import platform
import socket
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
VISUALS = EVIDENCE / "visuals"
REPORTS = ROOT / "reports"


class Config:
    vocab_size = 97
    d_model = 64
    n_heads = 4
    n_layers = 3
    max_seq_len = 256
    batch_size = 2
    prompt_len = 24
    decode_steps = 12
    seed = 20261114
    tolerance = 2e-5


def split_heads(x: torch.Tensor, heads: int):
    batch, seq_len, channels = x.shape
    return x.view(batch, seq_len, heads, channels // heads).transpose(1, 2)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)

    def project(self, x):
        batch, seq_len, channels = x.shape
        qkv = self.qkv(x).view(
            batch, seq_len, 3, self.heads, self.head_dim
        )
        return qkv.permute(2, 0, 3, 1, 4).unbind(0)

    def merge(self, x):
        batch, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(
            batch, seq_len, self.heads * self.head_dim
        )

    def prefill(self, x):
        # Prompt positions are evaluated together with a causal mask.
        q, k, v = self.project(x)
        length = x.size(1)
        future = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = F.softmax(scores.masked_fill(future, float("-inf")), dim=-1)
        output = self.proj(self.merge(weights @ v))
        return output, (k, v)

    def decode(self, x, past_kv):
        # The one-token query can attend to all cached past + current K/V.
        q, new_k, new_v = self.project(x)
        past_k, past_v = past_kv
        all_k = torch.cat((past_k, new_k), dim=-2)
        all_v = torch.cat((past_v, new_v), dim=-2)
        scores = (q @ all_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # No triangular mask here: all entries are past/current positions.
        # A naive non-square causal mask can be upper-left aligned and wrong.
        weights = F.softmax(scores, dim=-1)
        output = self.proj(self.merge(weights @ all_v))
        return output, (all_k, all_v)


class DecoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )

    def prefill(self, x):
        a, cache = self.attn.prefill(self.norm1(x))
        x = x + a
        return x + self.mlp(self.norm2(x)), cache

    def decode(self, x, cache):
        a, next_cache = self.attn.decode(self.norm1(x), cache)
        x = x + a
        return x + self.mlp(self.norm2(x)), next_cache


class TinyDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.token.weight

    def forward(self, input_ids):
        _, length = input_ids.shape
        if length > self.cfg.max_seq_len:
            raise ValueError("Sequence exceeds max_seq_len")
        pos = torch.arange(length, device=input_ids.device)
        x = self.token(input_ids) + self.position(pos)
        for block in self.blocks:
            x, _ = block.prefill(x)
        return self.head(self.norm(x))

    def prefill(self, input_ids):
        _, length = input_ids.shape
        if length > self.cfg.max_seq_len:
            raise ValueError("Prompt exceeds max_seq_len")
        pos = torch.arange(length, device=input_ids.device)
        x = self.token(input_ids) + self.position(pos)
        caches = []
        for block in self.blocks:
            x, cache = block.prefill(x)
            caches.append(cache)
        return self.head(self.norm(x))[:, -1, :], caches

    def decode(self, input_id, position, caches):
        if position >= self.cfg.max_seq_len:
            raise ValueError("Decode position exceeds max_seq_len")
        pos = torch.tensor([position], device=input_id.device)
        x = self.token(input_id) + self.position(pos)
        next_caches = []
        for block, cache in zip(self.blocks, caches):
            x, new_cache = block.decode(x, cache)
            next_caches.append(new_cache)
        return self.head(self.norm(x))[:, -1, :], next_caches


def cache_nbytes(caches):
    return sum(
        (k.numel() + v.numel()) * k.element_size()
        for k, v in caches
    )


def kv_bytes(layers, batch, tokens, kv_heads, head_dim, bytes_per_element):
    return 2 * layers * batch * tokens * kv_heads * head_dim * bytes_per_element


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def make_svg_flow():
    return """<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="520" viewBox="0 0 1200 520" role="img" aria-labelledby="t d">
<title id="t">Prefill and decode</title><desc id="d">Prefill evaluates prompt positions in parallel and seeds key and value caches. Decode evaluates one new token against the cached prefix.</desc>
<rect width="1200" height="520" fill="#fbfaf7"/>
<text x="48" y="48" font-family="Arial" font-size="24" font-weight="700" fill="#192630">Two phases, different query shapes</text>
<rect x="48" y="74" width="520" height="390" rx="7" fill="#e3ecec" stroke="#b8c9c9"/>
<rect x="632" y="74" width="520" height="390" rx="7" fill="#f3e9df" stroke="#d9c5b6"/>
<text x="78" y="116" font-family="Arial" font-size="21" font-weight="700" fill="#17394b">01 / PREFILL</text>
<text x="662" y="116" font-family="Arial" font-size="21" font-weight="700" fill="#7b412e">02 / DECODE</text>
<text x="78" y="144" font-family="Arial" font-size="14" fill="#34444d">All prompt positions processed together</text>
<text x="662" y="144" font-family="Arial" font-size="14" fill="#34444d">One new query attends to past and current K/V</text>
<g font-family="Arial" font-size="13" text-anchor="middle" fill="white">
<rect x="82" y="174" width="48" height="42" rx="4" fill="#28566b"/><text x="106" y="200">t0</text>
<rect x="142" y="174" width="48" height="42" rx="4" fill="#28566b"/><text x="166" y="200">t1</text>
<rect x="202" y="174" width="48" height="42" rx="4" fill="#28566b"/><text x="226" y="200">t2</text>
<rect x="262" y="174" width="48" height="42" rx="4" fill="#28566b"/><text x="286" y="200">t3</text>
<rect x="322" y="174" width="48" height="42" rx="4" fill="#28566b"/><text x="346" y="200">t4</text>
<rect x="382" y="174" width="48" height="42" rx="4" fill="#28566b"/><text x="406" y="200">t5</text>
<rect x="442" y="174" width="48" height="42" rx="4" fill="#28566b"/><text x="466" y="200">…</text>
</g>
<path d="M286 220 V248 M278 240 L286 250 L294 240" stroke="#28566b" stroke-width="3" fill="none"/>
<rect x="108" y="258" width="356" height="54" rx="5" fill="#17394b"/><text x="286" y="291" text-anchor="middle" font-family="Arial" font-size="17" fill="white">Transformer layers · Q / K / V</text>
<path d="M286 313 V338 M278 330 L286 340 L294 330" stroke="#28566b" stroke-width="3" fill="none"/>
<rect x="108" y="350" width="168" height="76" rx="5" fill="white" stroke="#28566b"/><text x="192" y="380" text-anchor="middle" font-family="Arial" font-size="16" font-weight="700" fill="#17394b">K cache</text><text x="192" y="404" text-anchor="middle" font-family="Arial" font-size="13" fill="#34444d">prompt positions</text>
<rect x="296" y="350" width="168" height="76" rx="5" fill="white" stroke="#28566b"/><text x="380" y="380" text-anchor="middle" font-family="Arial" font-size="16" font-weight="700" fill="#17394b">V cache</text><text x="380" y="404" text-anchor="middle" font-family="Arial" font-size="13" fill="#34444d">prompt positions</text>
<g font-family="Arial" font-size="13" text-anchor="middle" fill="white">
<rect x="672" y="174" width="64" height="42" rx="4" fill="#8e9b98"/><text x="704" y="200">past</text>
<rect x="748" y="174" width="64" height="42" rx="4" fill="#8e9b98"/><text x="780" y="200">past</text>
<rect x="824" y="174" width="64" height="42" rx="4" fill="#8e9b98"/><text x="856" y="200">past</text>
<rect x="900" y="174" width="64" height="42" rx="4" fill="#8e9b98"/><text x="932" y="200">past</text>
<rect x="976" y="174" width="64" height="42" rx="4" fill="#a9573d"/><text x="1008" y="200">new</text>
</g>
<path d="M892 220 V248 M884 240 L892 250 L900 240" stroke="#a9573d" stroke-width="3" fill="none"/>
<rect x="688" y="258" width="408" height="54" rx="5" fill="#6c4032"/><text x="892" y="291" text-anchor="middle" font-family="Arial" font-size="16" fill="white">Q(new) × K(all) → softmax → V(all)</text>
<path d="M892 313 V338 M884 330 L892 340 L900 330" stroke="#a9573d" stroke-width="3" fill="none"/>
<rect x="690" y="350" width="190" height="76" rx="5" fill="white" stroke="#a9573d"/><text x="785" y="380" text-anchor="middle" font-family="Arial" font-size="15" font-weight="700" fill="#6c4032">Append K(new)</text><text x="785" y="404" text-anchor="middle" font-family="Arial" font-size="13" fill="#34444d">at every layer</text>
<rect x="904" y="350" width="190" height="76" rx="5" fill="white" stroke="#a9573d"/><text x="999" y="380" text-anchor="middle" font-family="Arial" font-size="15" font-weight="700" fill="#6c4032">Append V(new)</text><text x="999" y="404" text-anchor="middle" font-family="Arial" font-size="13" fill="#34444d">at every layer</text>
<text x="600" y="493" text-anchor="middle" font-family="Arial" font-size="13" fill="#647078">Conceptual dataflow · cache is maintained independently at each attention layer</text></svg>"""


def make_svg_memory():
    rows = []
    for i, tokens in enumerate((1024, 2048, 4096, 8192, 16384, 32768)):
        size = kv_bytes(32, 1, tokens, 8, 128, 2)
        mib = size / (1024 ** 2)
        width = 600 * tokens / 32768
        label = f"{mib:,.0f} MiB" if mib < 1024 else f"{mib/1024:.1f} GiB"
        y = 122 + i * 53
        rows.append(
            f'<text x="48" y="{y+20}" font-family="Arial" font-size="14" fill="#34444d">{tokens:,} tokens</text>'
            f'<rect x="190" y="{y}" width="{width:.1f}" height="28" rx="3" fill="#28566b"/>'
            f'<text x="{202+width:.1f}" y="{y+19}" font-family="Arial" font-size="14" fill="#192630">{label}</text>'
        )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="500" viewBox="0 0 1000 500" role="img" aria-labelledby="t d">'
        '<title id="t">KV cache memory grows with sequence length</title>'
        '<desc id="d">FP16 or BF16 illustrative cache size for 32 layers, eight KV heads and head dimension 128.</desc>'
        '<rect width="1000" height="500" fill="#fbfaf7"/>'
        '<text x="48" y="52" font-family="Arial" font-size="23" font-weight="700" fill="#192630">KV state grows linearly with cached tokens</text>'
        '<text x="48" y="82" font-family="Arial" font-size="14" fill="#647078">32 layers · batch 1 · 8 KV heads · head_dim 128 · 2 bytes/value</text>'
        + ''.join(rows) +
        '<text x="48" y="465" font-family="Arial" font-size="13" fill="#647078">Excludes allocator metadata, fragmentation, workspaces, weights and activations.</text></svg>'
    )


def make_svg_work():
    prompts = 32
    rows = []
    data = []
    for i, steps in enumerate((4, 8, 16, 32, 64)):
        cached = prompts + steps
        repeated = sum(prompts + j + 1 for j in range(steps))
        data.append((steps, cached, repeated))
    maximum = max(x[2] for x in data)
    for i, (steps, cached, repeated) in enumerate(data):
        y = 132 + i * 61
        w1 = 540 * repeated / maximum
        w2 = 540 * cached / maximum
        rows.append(
            f'<text x="48" y="{y+17}" font-family="Arial" font-size="14" fill="#34444d">D={steps}</text>'
            f'<rect x="130" y="{y}" width="{w1:.1f}" height="20" rx="3" fill="#a9573d"/>'
            f'<rect x="130" y="{y+25}" width="{w2:.1f}" height="20" rx="3" fill="#28566b"/>'
            f'<text x="{140+w1:.1f}" y="{y+16}" font-family="Arial" font-size="12" fill="#192630">{repeated}</text>'
            f'<text x="{140+w2:.1f}" y="{y+41}" font-family="Arial" font-size="12" fill="#192630">{cached}</text>'
        )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="500" viewBox="0 0 1000 500" role="img" aria-labelledby="t d">'
        '<title id="t">Input positions with and without KV cache</title>'
        '<desc id="d">For a fixed 32-token prompt, cached decoding processes the prompt once and then one new token per step; full-prefix recomputation processes every prefix again.</desc>'
        '<rect width="1000" height="500" fill="#fbfaf7"/>'
        '<text x="48" y="52" font-family="Arial" font-size="23" font-weight="700" fill="#192630">Avoid reprocessing the entire prefix at every step</text>'
        '<rect x="48" y="80" width="14" height="14" fill="#a9573d"/><text x="70" y="93" font-family="Arial" font-size="13" fill="#34444d">Full-prefix recomputation</text>'
        '<rect x="280" y="80" width="14" height="14" fill="#28566b"/><text x="302" y="93" font-family="Arial" font-size="13" fill="#34444d">Prefill once + incremental decode</text>'
        + ''.join(rows) +
        '<text x="48" y="465" font-family="Arial" font-size="13" fill="#647078">Structural input-position count, not FLOPs, latency or measured speedup.</text></svg>'
    )


def main():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    VISUALS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(Config.seed)
    cfg = Config()
    model = TinyDecoder(cfg).cpu().eval()

    generator = torch.Generator().manual_seed(cfg.seed + 1)
    sequence = torch.randint(
        0, cfg.vocab_size,
        (cfg.batch_size, cfg.prompt_len + cfg.decode_steps),
        generator=generator,
    )
    prompt = sequence[:, :cfg.prompt_len]
    continuation = sequence[:, cfg.prompt_len:]

    evidence = []
    lengths = []
    with torch.inference_mode():
        cached_logits, caches = model.prefill(prompt)
        full_logits = model(prompt)[:, -1, :]
        prefill_error = float((cached_logits - full_logits).abs().max().item())
        evidence.append(["prefill", 0, cfg.prompt_len, prefill_error])
        lengths.append(cfg.prompt_len)

        for step in range(cfg.decode_steps):
            token = continuation[:, step:step + 1]
            cached_logits, caches = model.decode(
                token, cfg.prompt_len + step, caches
            )
            prefix = torch.cat((prompt, continuation[:, :step + 1]), dim=1)
            reference = model(prefix)[:, -1, :]
            error = float((cached_logits - reference).abs().max().item())
            length = cfg.prompt_len + step + 1
            evidence.append(["decode", step + 1, length, error])
            lengths.append(length)

    max_error = max(row[3] for row in evidence)
    final_len = cfg.prompt_len + cfg.decode_steps
    measured_bytes = cache_nbytes(caches)
    formula_bytes = kv_bytes(
        cfg.n_layers, cfg.batch_size, final_len,
        cfg.n_heads, cfg.d_model // cfg.n_heads, 4,
    )
    cached_positions = cfg.prompt_len + cfg.decode_steps
    recomputed_positions = sum(
        cfg.prompt_len + i + 1 for i in range(cfg.decode_steps)
    )
    reduction = 100 * (1 - cached_positions / recomputed_positions)

    checks = {
        "prefill_logits_match_full_forward": prefill_error < cfg.tolerance,
        "all_decode_logits_match_full_recompute": max_error < cfg.tolerance,
        "cache_length_increments_once_per_token": (
            lengths == list(range(cfg.prompt_len, final_len + 1))
        ),
        "all_layers_have_expected_cache_shape": all(
            k.shape == (
                cfg.batch_size, cfg.n_heads, final_len,
                cfg.d_model // cfg.n_heads,
            ) and v.shape == k.shape
            for k, v in caches
        ),
        "measured_cache_bytes_match_formula": measured_bytes == formula_bytes,
        "cached_path_processes_fewer_token_positions": (
            cached_positions < recomputed_positions
        ),
        "logit_errors_are_finite": all(math.isfinite(row[3]) for row in evidence),
    }
    ok = all(checks.values())

    write_csv(
        EVIDENCE / "cache-equivalence.csv",
        ["stage", "step", "cache_length", "max_abs_logit_error"],
        [[a, b, c, f"{d:.12g}"] for a, b, c, d in evidence],
    )
    pos_rows = []
    for steps in (4, 8, 16, 32, 64):
        cached = 32 + steps
        repeated = sum(32 + i + 1 for i in range(steps))
        pos_rows.append([32, steps, cached, repeated, f"{100*(1-cached/repeated):.2f}"])
    write_csv(
        EVIDENCE / "token-position-comparison.csv",
        ["prompt_tokens", "decode_inputs", "cached_positions",
         "full_prefix_recompute_positions", "structural_reduction_percent"],
        pos_rows,
    )
    memory_rows = []
    for tokens in (1024, 2048, 4096, 8192, 16384, 32768):
        size = kv_bytes(32, 1, tokens, 8, 128, 2)
        memory_rows.append([tokens, 32, 1, 8, 128, 2, size, size / 1024**2])
    write_csv(
        EVIDENCE / "kv-cache-memory.csv",
        ["tokens", "layers", "batch", "kv_heads", "head_dim",
         "bytes_per_element", "cache_bytes", "cache_mib"],
        memory_rows,
    )

    metrics = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "device": "CPU",
        "dtype": "float32",
        "cuda_available": torch.cuda.is_available(),
        "configuration": {
            "vocab_size": cfg.vocab_size, "d_model": cfg.d_model,
            "layers": cfg.n_layers, "query_heads": cfg.n_heads,
            "kv_heads": cfg.n_heads, "head_dim": cfg.d_model // cfg.n_heads,
            "batch_size": cfg.batch_size, "prompt_tokens": cfg.prompt_len,
            "decode_inputs": cfg.decode_steps, "seed": cfg.seed,
        },
        "results": {
            "prefill_max_abs_logit_error": prefill_error,
            "decode_max_abs_logit_error": max_error,
            "steps_checked": cfg.decode_steps,
            "final_cache_length": final_len,
            "measured_cache_bytes": measured_bytes,
            "formula_cache_bytes": formula_bytes,
            "cached_positions": cached_positions,
            "full_prefix_recompute_positions": recomputed_positions,
            "structural_reduction_percent": reduction,
        },
        "checks": checks,
        "step_evidence": [
            {"stage": a, "step": b, "cache_length": c, "max_abs_logit_error": d}
            for a, b, c, d in evidence
        ],
        "limitations": [
            "Randomly initialised toy model; no language quality claim.",
            "CPU correctness test, not a GPU performance benchmark.",
            "Structural token-position count is not a FLOP or latency measurement.",
            "No PagedAttention, prefix sharing, quantised cache or distributed serving.",
        ],
    }
    (REPORTS / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (VISUALS / "prefill-decode-flow.svg").write_text(make_svg_flow(), encoding="utf-8")
    (VISUALS / "kv-cache-memory.svg").write_text(make_svg_memory(), encoding="utf-8")
    (VISUALS / "token-position-comparison.svg").write_text(make_svg_work(), encoding="utf-8")

    # 16,384 tokens, 32 layers, batch 1, 8 KV heads, dim 128, 2 bytes/value = 2 GiB.
    example_bytes = kv_bytes(32, 1, 16384, 8, 128, 2)
    report = f"""# Lab report — Prefill, Decode and KV Cache

- Host: {socket.gethostname()}
- Python: {platform.python_version()}
- PyTorch: {torch.__version__}
- Device: CPU · float32 · 4 configured CPU threads
- Model: {cfg.n_layers} decoder blocks · {cfg.n_heads} heads · d_model {cfg.d_model}
- Batch: {cfg.batch_size}
- Prompt: {cfg.prompt_len} token positions
- Incremental decode inputs: {cfg.decode_steps}
- Seed: {cfg.seed}

## Numerical equivalence

- Prefill vs full-forward max absolute logit error: {prefill_error:.12g}
- Maximum error across incremental decode: {max_error:.12g}
- Comparisons: one prefill plus {cfg.decode_steps} incremental steps.
- Each incremental output is compared with the last-position logits from a full-prefix forward pass.

## KV cache accounting

- Final sequence length: {final_len}
- Measured K/V tensor storage: {measured_bytes:,} bytes
- Formula result: {formula_bytes:,} bytes
- Cached-path model input positions: {cached_positions}
- Recomputed full-prefix model input positions: {recomputed_positions}
- Structural reduction in input positions: {reduction:.2f}%

## Illustrative production-sized memory calculation

For 32 layers, batch 1, 8 KV heads, head_dim 128 and 2 bytes per value:
- 16,384 cached tokens: {example_bytes:,} bytes ({example_bytes/1024**3:.2f} GiB).
- Formula: 2 × layers × batch × tokens × KV heads × head_dim × bytes_per_element.

## Automated checks

"""
    for name, passed in checks.items():
        report += f"- {name}: {'PASS' if passed else 'FAIL'}\n"
    report += f"\n**LAB STATUS: {'PASS' if ok else 'FAIL'}**\n\n"
    report += """## Boundaries

- The model is randomly initialised and validates inference mechanics only. It does not generate meaningful language and is not a trained LLM.
- This CPU run validates numerical equivalence and cache accounting. It makes no GPU acceleration, throughput or latency claim.
- The structural count does not equal FLOPs: the new query still reads the accumulated K/V context at each decode step.
- The memory example assumes full attention with the stated KV head count and dtype. It excludes allocator fragmentation, metadata, workspaces, weights and activations.
- No model weights, CUDA runtime, external dataset, hosted endpoint, API or paid service is downloaded or required.
"""
    (REPORTS / "lab-report.md").write_text(report, encoding="utf-8")

    print("=== INFERENCE LAB SUMMARY ===")
    print(f"host={socket.gethostname()} python={platform.python_version()} torch={torch.__version__} device=cpu")
    print(f"prefill_error={prefill_error:.12g} decode_max_error={max_error:.12g}")
    print(f"cache_bytes={measured_bytes} formula_bytes={formula_bytes}")
    print(f"cached_positions={cached_positions} full_prefix_positions={recomputed_positions}")
    print(f"structural_reduction_percent={reduction:.2f}")
    print("checks=" + json.dumps(checks, ensure_ascii=False))
    print("LAB STATUS: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
