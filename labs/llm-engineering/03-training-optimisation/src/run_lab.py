#!/usr/bin/env python
"""Training and optimisation lab for Consola Abierta, chapter 03.

A small character-level decoder-only Transformer. CPU-only, deterministic
batch plan, two AdamW learning-rate policies and explicit numerical checks.
This is a teaching experiment, not a production language model benchmark.
"""
from __future__ import annotations

import csv
import json
import math
import os
import platform
import random
import socket
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "corpus_es.txt"
EVIDENCE = ROOT / "evidence"
VISUALS = EVIDENCE / "visuals"
REPORTS = ROOT / "reports"


class Config:
    block_size = 64
    d_model = 48
    n_heads = 4
    n_layers = 2
    dropout = 0.05
    batch_size = 12
    steps = 240
    eval_interval = 40
    eval_batches = 8
    peak_lr = 5e-4
    min_lr = 5e-5
    warmup_steps = 24
    weight_decay = 0.01
    max_grad_norm = 1.0
    seed = 20261115


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        if cfg.d_model % cfg.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_dropout = cfg.dropout
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_dropout(self.proj(out))


class DecoderBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class TinyDecoder(nn.Module):
    def __init__(self, vocab_size: int, cfg: Config):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, cfg.d_model)
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.Sequential(*[DecoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, t = input_ids.shape
        if t > self.position_embedding.num_embeddings:
            raise ValueError("Input sequence exceeds block_size")
        positions = torch.arange(t, device=input_ids.device)
        x = self.dropout(self.token_embedding(input_ids) + self.position_embedding(positions))
        x = self.blocks(x)
        return self.lm_head(self.final_norm(x))


def load_data():
    raw = DATA_PATH.read_text(encoding="utf-8")
    if "===TRAIN===" not in raw or "===VALIDATION===" not in raw:
        raise ValueError("Expected ===TRAIN=== and ===VALIDATION=== corpus sections")
    train_text = raw.split("===TRAIN===", 1)[1].split("===VALIDATION===", 1)[0].strip()
    val_text = raw.split("===VALIDATION===", 1)[1].strip()
    if len(train_text) <= Config.block_size + 1 or len(val_text) <= Config.block_size + 1:
        raise ValueError("Train/validation split is too short")
    alphabet = sorted(set(train_text + val_text))
    stoi = {ch: i for i, ch in enumerate(alphabet)}
    encode = lambda text: torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    return train_text, val_text, alphabet, encode(train_text), encode(val_text)


def make_batches(data: torch.Tensor, count: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    high = data.numel() - Config.block_size
    starts = torch.randint(0, high, (count, Config.batch_size), generator=generator)
    offsets = torch.arange(Config.block_size)
    batches = []
    for row in starts:
        positions = row[:, None] + offsets[None, :]
        x = data[positions]
        y = data[positions + 1]
        batches.append((x, y))
    return batches


def next_token_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def schedule_lr(step_index: int) -> float:
    if step_index < Config.warmup_steps:
        return Config.peak_lr * (step_index + 1) / Config.warmup_steps
    decay_steps = max(1, Config.steps - Config.warmup_steps - 1)
    progress = min(1.0, (step_index - Config.warmup_steps) / decay_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return Config.min_lr + (Config.peak_lr - Config.min_lr) * cosine


@torch.no_grad()
def evaluate(model: nn.Module, batches) -> float:
    was_training = model.training
    model.eval()
    values = []
    for x, y in batches:
        logits = model(x)
        values.append(float(next_token_loss(logits, y).item()))
    model.train(was_training)
    return sum(values) / len(values)


def optimizer_for(model: nn.Module):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim >= 2 and "embedding" not in name:
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": Config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=Config.peak_lr, betas=(0.9, 0.95), eps=1e-8,
    )


def numerical_checks():
    # The targets must be the input sequence shifted one token to the left.
    sample = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    x, y = sample[:-1], sample[1:]
    shift_ok = bool(torch.equal(y[:-1], x[1:]) and x.numel() == y.numel())

    # Cross-entropy on logits must match log_softmax followed by NLL.
    torch.manual_seed(Config.seed + 10)
    logits = torch.randn(2, 3, 7, dtype=torch.float64, requires_grad=True)
    targets = torch.randint(0, 7, (2, 3))
    flat_logits, flat_targets = logits.reshape(-1, 7), targets.reshape(-1)
    ce = F.cross_entropy(flat_logits, flat_targets)
    nll = F.nll_loss(F.log_softmax(flat_logits, dim=-1), flat_targets)
    grad = torch.autograd.grad(ce, logits)[0].reshape(-1, 7)
    probs = flat_logits.detach().softmax(dim=-1)
    manual_grad = probs.clone()
    manual_grad[torch.arange(flat_targets.numel()), flat_targets] -= 1.0
    manual_grad /= flat_targets.numel()
    grad_error = float((grad - manual_grad).abs().max().item())
    ce_error = float((ce - nll).abs().item())

    # AdamW applies decoupled decay even when the loss gradient is zero.
    parameter = nn.Parameter(torch.tensor([1.0]))
    test_optimizer = torch.optim.AdamW(
        [parameter], lr=0.1, betas=(0.0, 0.0), weight_decay=0.2
    )
    parameter.grad = torch.zeros_like(parameter)
    test_optimizer.step()
    adamw_expected = 0.98
    adamw_error = abs(float(parameter.item()) - adamw_expected)

    # Gradient clipping scales down, never amplifies, the global norm.
    clipped = nn.Parameter(torch.tensor([3.0, 4.0]))
    clipped.grad = torch.tensor([3.0, 4.0])
    before_clip = float(torch.nn.utils.clip_grad_norm_([clipped], 1.0).item())
    after_clip = float(clipped.grad.norm().item())

    schedule = [schedule_lr(i) for i in range(Config.steps)]
    warmup_start_ok = schedule[0] > 0
    warmup_peak_ok = abs(schedule[Config.warmup_steps - 1] - Config.peak_lr) < 1e-12
    cosine_min_ok = abs(schedule[-1] - Config.min_lr) < 1e-12
    cosine_monotonic_ok = all(
        schedule[i + 1] <= schedule[i] + 1e-12
        for i in range(Config.warmup_steps, len(schedule) - 1)
    )
    return {
        "shift_alignment_check": shift_ok,
        "cross_entropy_vs_log_softmax_nll_max_abs_error": ce_error,
        "cross_entropy_equivalence_check": ce_error < 1e-12,
        "cross_entropy_gradient_max_abs_error": grad_error,
        "cross_entropy_gradient_check": grad_error < 1e-12,
        "adamw_zero_gradient_expected_parameter": adamw_expected,
        "adamw_zero_gradient_actual_parameter": float(parameter.item()),
        "adamw_decoupled_decay_check": adamw_error < 1e-7,
        "gradient_norm_before_clip": before_clip,
        "gradient_norm_after_clip": after_clip,
        "gradient_clipping_check": before_clip > 1.0 and after_clip <= 1.0 + 1e-7,
        "warmup_start_positive_check": bool(warmup_start_ok),
        "warmup_peak_reached_check": bool(warmup_peak_ok),
        "cosine_minimum_reached_check": bool(cosine_min_ok),
        "cosine_decay_monotonic_check": bool(cosine_monotonic_ok),
    }


def run_training(initial_state, train_batches, train_eval, val_eval, mode: str):
    torch.manual_seed(Config.seed + 77)
    model = TinyDecoder(VOCAB_SIZE, Config)
    model.load_state_dict(initial_state)
    optimizer = optimizer_for(model)
    history = []
    rows = []
    initial = {
        "train_loss": evaluate(model, train_eval),
        "validation_loss": evaluate(model, val_eval),
    }
    history.append({"step": 0, **initial})
    start = time.perf_counter()
    model.train()
    for step, (x, y) in enumerate(train_batches):
        lr = Config.peak_lr if mode == "constant" else schedule_lr(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = next_token_loss(logits, y)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), Config.max_grad_norm, error_if_nonfinite=True
        ).item())
        optimizer.step()
        rows.append({
            "optimizer": mode, "step": step + 1, "learning_rate": lr,
            "batch_loss": float(loss.item()), "gradient_norm_preclip": grad_norm,
        })
        if (step + 1) % Config.eval_interval == 0 or step + 1 == Config.steps:
            losses = {
                "train_loss": evaluate(model, train_eval),
                "validation_loss": evaluate(model, val_eval),
            }
            history.append({"step": step + 1, **losses})
            print(
                f"{mode:15s} step={step + 1:03d}/{Config.steps} "
                f"train={losses['train_loss']:.4f} "
                f"validation={losses['validation_loss']:.4f} "
                f"lr={lr:.2e} grad_norm={grad_norm:.4f}"
            )
    elapsed = time.perf_counter() - start
    final = history[-1]
    model.train(False)
    return {
        "mode": mode, "initial": initial, "final": final,
        "history": history, "rows": rows, "wall_time_seconds": elapsed,
        "model_parameters": sum(p.numel() for p in model.parameters()),
    }


def write_loss_svg(results, path: Path):
    width, height = 1200, 660
    left, right, top, bottom = 100, 60, 80, 100
    plot_w, plot_h = width - left - right, height - top - bottom
    values = [row[key] for result in results for row in result["history"]
              for key in ("train_loss", "validation_loss")]
    y_min = max(0.0, min(values) * 0.9)
    y_max = max(values) * 1.06
    if y_max - y_min < 0.1:
        y_max = y_min + 0.1

    def px(step):
        return left + plot_w * step / Config.steps

    def py(value):
        return top + plot_h * (y_max - value) / (y_max - y_min)

    colors = {
        ("constant", "train_loss"): "#2fd7f2",
        ("constant", "validation_loss"): "#f6a35b",
        ("warmup_cosine", "train_loss"): "#a78bfa",
        ("warmup_cosine", "validation_loss"): "#8bd49c",
    }
    elements = []
    for i in range(6):
        value = y_min + (y_max - y_min) * i / 5
        yy = py(value)
        elements.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#24344a"/>')
        elements.append(f'<text x="{left-14}" y="{yy+5:.1f}" text-anchor="end" fill="#9aabc2" font-size="16">{value:.2f}</text>')
    for i in range(7):
        step = Config.steps * i / 6
        xx = px(step)
        elements.append(f'<text x="{xx:.1f}" y="{height-55}" text-anchor="middle" fill="#9aabc2" font-size="15">{int(step)}</text>')
    legends = []
    legend_items = [
        ("constant", "train_loss", "Constant · train"),
        ("constant", "validation_loss", "Constant · validation"),
        ("warmup_cosine", "train_loss", "Warmup + cosine · train"),
        ("warmup_cosine", "validation_loss", "Warmup + cosine · validation"),
    ]
    for idx, (mode, key, label) in enumerate(legend_items):
        x = 120 + (idx % 2) * 470
        y = 35 + (idx // 2) * 24
        color = colors[(mode, key)]
        legends.append(f'<line x1="{x}" y1="{y}" x2="{x+30}" y2="{y}" stroke="{color}" stroke-width="4"/><text x="{x+40}" y="{y+5}" fill="#d6e2f0" font-size="16">{label}</text>')
    for result in results:
        for key in ("train_loss", "validation_loss"):
            points = " ".join(
                f'{px(row["step"]):.1f},{py(row[key]):.1f}' for row in result["history"]
            )
            color = colors[(result["mode"], key)]
            elements.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#08111f"/>
<text x="{left}" y="25" fill="#eef6ff" font-size="23" font-weight="700">Training policy comparison · next-token cross entropy</text>
{"".join(legends)}
{"".join(elements)}
<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#8091a8"/>
<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#8091a8"/>
<text x="{left+8}" y="{height-17}" fill="#9aabc2" font-size="15">Optimizer updates</text>
</svg>'''
    path.write_text(svg, encoding="utf-8")


def write_lr_svg(path: Path):
    values = [(i, schedule_lr(i)) for i in range(Config.steps)]
    width, height = 1100, 420
    left, right, top, bottom = 80, 40, 50, 70
    points = " ".join(
        f'{left + (width-left-right)*i/ (Config.steps-1):.1f},'
        f'{top + (height-top-bottom)*(1-lr/Config.peak_lr):.1f}'
        for i, lr in values
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#08111f"/>
<text x="{left}" y="30" fill="#eef6ff" font-size="22" font-weight="700">Warmup + cosine learning-rate schedule</text>
<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#8091a8"/>
<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#8091a8"/>
<polyline points="{points}" fill="none" stroke="#2fd7f2" stroke-width="4" stroke-linecap="round"/>
<line x1="{left + (width-left-right)*Config.warmup_steps/Config.steps:.1f}" y1="{top}" x2="{left + (width-left-right)*Config.warmup_steps/Config.steps:.1f}" y2="{height-bottom}" stroke="#f6a35b" stroke-dasharray="6 6"/>
<text x="{left+8}" y="{height-20}" fill="#9aabc2" font-size="15">Step 0 → {Config.steps-1}</text>
<text x="{left + (width-left-right)*Config.warmup_steps/Config.steps + 8:.1f}" y="{top+22}" fill="#f6a35b" font-size="14">Warmup ends</text>
<text x="{left-8}" y="{top+5}" text-anchor="end" fill="#9aabc2" font-size="14">{Config.peak_lr:.1e}</text>
<text x="{left-8}" y="{height-bottom}" text-anchor="end" fill="#9aabc2" font-size="14">{Config.min_lr:.1e}</text>
</svg>'''
    path.write_text(svg, encoding="utf-8")


def main():
    global VOCAB_SIZE
    for folder in (EVIDENCE, VISUALS, REPORTS):
        folder.mkdir(parents=True, exist_ok=True)
    random.seed(Config.seed)
    torch.manual_seed(Config.seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = torch.device("cpu")

    train_text, val_text, alphabet, train_ids, val_ids = load_data()
    VOCAB_SIZE = len(alphabet)
    train_batches = make_batches(train_ids, Config.steps, Config.seed + 1)
    train_eval = make_batches(train_ids, Config.eval_batches, Config.seed + 2)
    val_eval = make_batches(val_ids, Config.eval_batches, Config.seed + 3)

    template = TinyDecoder(VOCAB_SIZE, Config)
    initial_state = deepcopy(template.state_dict())
    del template

    checks = numerical_checks()
    results = [
        run_training(initial_state, train_batches, train_eval, val_eval, "constant"),
        run_training(initial_state, train_batches, train_eval, val_eval, "warmup_cosine"),
    ]
    for result in results:
        result["train_loss_reduction"] = (
            result["initial"]["train_loss"] - result["final"]["train_loss"]
        )
        result["validation_loss_reduction"] = (
            result["initial"]["validation_loss"] - result["final"]["validation_loss"]
        )
    checks["both_policies_reduce_train_loss_check"] = all(
        r["train_loss_reduction"] > 0 for r in results
    )
    checks["both_policies_reduce_validation_loss_check"] = all(
        r["validation_loss_reduction"] > 0 for r in results
    )
    checks["all_final_losses_finite_check"] = all(
        math.isfinite(r["final"]["train_loss"]) and math.isfinite(r["final"]["validation_loss"])
        for r in results
    )

    with (EVIDENCE / "training-comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "optimizer", "step", "learning_rate", "batch_loss", "gradient_norm_preclip"
        ])
        writer.writeheader()
        for result in results:
            writer.writerows(result["rows"])

    history = {r["mode"]: r["history"] for r in results}
    (EVIDENCE / "loss-history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_loss_svg(results, VISUALS / "training-policy-comparison.svg")
    write_lr_svg(VISUALS / "warmup-cosine-schedule.svg")

    metrics = {
        "lab": "Consola Abierta - How an LLM Learns 03",
        "run_timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "device": str(device),
        "cpu_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "xpu_available": bool(getattr(torch, "xpu", None) and torch.xpu.is_available()),
        "data": {
            "source": "Chapter 01 authored Spanish teaching corpus",
            "train_characters": len(train_text),
            "validation_characters": len(val_text),
            "vocabulary_characters": VOCAB_SIZE,
            "tokenization": "character-level teaching tokenizer",
            "validation_used_for_optimizer_updates": False,
        },
        "model": {
            "architecture": "2-layer Pre-LN decoder-only Transformer with causal SDPA",
            "block_size": Config.block_size,
            "d_model": Config.d_model,
            "heads": Config.n_heads,
            "parameters": results[0]["model_parameters"],
            "batch_size": Config.batch_size,
            "dropout": Config.dropout,
            "trainable_parameters_shared_between_runs": True,
        },
        "training": {
            "steps_per_run": Config.steps,
            "seed": Config.seed,
            "optimizer": "AdamW, betas=(0.9, 0.95), eps=1e-8",
            "weight_decay": Config.weight_decay,
            "max_grad_norm": Config.max_grad_norm,
            "peak_learning_rate": Config.peak_lr,
            "minimum_learning_rate": Config.min_lr,
            "warmup_steps": Config.warmup_steps,
            "runs": [{
                "mode": r["mode"],
                "initial_train_loss": r["initial"]["train_loss"],
                "final_train_loss": r["final"]["train_loss"],
                "initial_validation_loss": r["initial"]["validation_loss"],
                "final_validation_loss": r["final"]["validation_loss"],
                "train_loss_reduction": r["train_loss_reduction"],
                "validation_loss_reduction": r["validation_loss_reduction"],
                "train_validation_gap_final": r["final"]["validation_loss"] - r["final"]["train_loss"],
                "wall_time_seconds": round(r["wall_time_seconds"], 3),
            } for r in results],
        },
        "checks": checks,
        "interpretation": [
            "The two runs start from identical weights and consume the same precomputed batches.",
            "This controlled comparison isolates the learning-rate policy within this small experiment; it does not prove one schedule is universally superior.",
            "The corpus is a small authored character-level teaching corpus, not a representative pretraining dataset.",
            "Validation loss is measured but never used for gradient updates.",
            "No GPU, API, hosted model, external dataset or paid service is used.",
        ],
    }
    (REPORTS / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = [
        "# Lab report — How an LLM Learns",
        "",
        f"- Host: {metrics['host']}",
        f"- Python: {metrics['python']}",
        f"- PyTorch: {metrics['pytorch']}",
        f"- Device: CPU ({metrics['cpu_threads']} threads)",
        f"- Vocabulary: {VOCAB_SIZE} characters",
        f"- Train / validation characters: {len(train_text):,} / {len(val_text):,}",
        f"- Parameters: {results[0]['model_parameters']:,}",
        f"- Steps per run: {Config.steps}",
        "",
        "## Controlled comparison",
        "",
        "| Policy | Train loss | Validation loss | Seconds |",
        "|---|---:|---:|---:|",
    ]
    for r in results:
        report.append(
            f"| {r['mode']} | {r['initial']['train_loss']:.4f} → {r['final']['train_loss']:.4f} "
            f"| {r['initial']['validation_loss']:.4f} → {r['final']['validation_loss']:.4f} "
            f"| {r['wall_time_seconds']:.3f} |"
        )
    report.extend(["", "## Numerical checks", ""])
    report.extend(f"- {key}: {value}" for key, value in checks.items())
    report.extend(["", "## Boundaries", ""])
    report.extend(f"- {item}" for item in metrics["interpretation"])
    (REPORTS / "lab-report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\n=== NUMERICAL CHECKS ===")
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    print("\n=== CONTROLLED RUNS ===")
    for r in results:
        print(
            f"{r['mode']}: train {r['initial']['train_loss']:.4f} -> {r['final']['train_loss']:.4f}; "
            f"validation {r['initial']['validation_loss']:.4f} -> {r['final']['validation_loss']:.4f}; "
            f"{r['wall_time_seconds']:.3f}s"
        )
    boolean_checks = [value for key, value in checks.items() if key.endswith("_check")]
    lab_ok = len(boolean_checks) == 12 and all(bool(value) for value in boolean_checks)
    print("\nLAB STATUS: " + ("PASS" if lab_ok else "FAIL"))
    return 0 if lab_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
