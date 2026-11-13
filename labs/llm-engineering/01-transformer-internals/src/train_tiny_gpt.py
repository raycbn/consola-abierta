#!/usr/bin/env python
"""Tiny decoder-only Transformer lab for Consola Abierta.

This is a CPU-sized pedagogical model, not a production LLM.
It uses a character tokenizer to expose tensor mechanics.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import socket
import time
import xml.sax.saxutils as sax
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "corpus_es.txt"
EVIDENCE = ROOT / "evidence"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"


@dataclass
class Config:
    block_size: int = 96
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.10
    batch_size: int = 16
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    steps: int = 600
    eval_interval: int = 50
    eval_batches: int = 20
    seed: int = 20261114


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with inspectable attention weights."""

    def __init__(self, config: Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        batch, seq_len, channels = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(batch, seq_len, 3, self.n_head, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        allowed = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(~allowed, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        used_weights = self.attn_dropout(weights)
        out = used_weights @ v
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        out = self.resid_dropout(self.proj(out))

        if return_weights:
            return out, weights
        return out, None


class FeedForward(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        inner = 4 * config.n_embd
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, inner),
            nn.GELU(),
            nn.Linear(inner, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    """GPT-style Pre-LayerNorm residual block."""

    def __init__(self, config: Config):
        super().__init__()
        self.ln_attn = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_mlp = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        attn_out, weights = self.attn(self.ln_attn(x), return_weights=return_weights)
        x = x + attn_out
        x = x + self.mlp(self.ln_mlp(x))
        return x, weights


class TinyGPT(nn.Module):
    """Small character-level decoder-only Transformer."""

    def __init__(self, vocab_size: int, config: Config):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)])
        self.final_norm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=False)
        # Weight tying: the output projection reuses the token embedding matrix.
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                return_attentions: bool = False):
        batch, seq_len = idx.shape
        if seq_len > self.config.block_size:
            raise ValueError(f"Sequence length {seq_len} exceeds block_size {self.config.block_size}")
        positions = torch.arange(seq_len, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)[None, :, :]
        x = self.dropout(x)
        all_attentions = []
        for block in self.blocks:
            x, weights = block(x, return_weights=return_attentions)
            if return_attentions:
                all_attentions.append(weights)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        if return_attentions:
            return logits, loss, all_attentions
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 200,
                 temperature: float = 0.8, top_k: int = 20):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None and top_k < logits.size(-1):
                cutoff = torch.topk(logits, top_k, dim=-1).values[:, [-1]]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx


def load_corpus(path: Path):
    text = path.read_text(encoding="utf-8")
    if "===TRAIN===" not in text or "===VALIDATION===" not in text:
        raise ValueError("Corpus must contain ===TRAIN=== and ===VALIDATION=== sections.")
    train_text = text.split("===TRAIN===", 1)[1].split("===VALIDATION===", 1)[0].strip()
    val_text = text.split("===VALIDATION===", 1)[1].strip()
    if len(train_text) < 1000 or len(val_text) < 300:
        raise ValueError("Corpus too small for the configured train/validation windows.")
    # The vocabulary is the union of observed characters. Validation examples
    # are not used for gradient updates.
    alphabet = sorted(set(train_text + val_text))
    stoi = {ch: i for i, ch in enumerate(alphabet)}
    itos = {i: ch for ch, i in stoi.items()}
    encode = lambda s: torch.tensor([stoi[c] for c in s], dtype=torch.long)
    train_ids = encode(train_text)
    val_ids = encode(val_text)
    return train_text, val_text, alphabet, stoi, itos, train_ids, val_ids


def get_batch(data: torch.Tensor, config: Config, device: torch.device):
    if data.numel() <= config.block_size + 1:
        raise ValueError("Data split is shorter than block_size + 1.")
    starts = torch.randint(0, data.numel() - config.block_size - 1,
                          (config.batch_size,))
    x = torch.stack([data[i:i + config.block_size] for i in starts])
    y = torch.stack([data[i + 1:i + config.block_size + 1] for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_ids, val_ids, config, device):
    model.eval()
    result = {}
    for split, data in (("train", train_ids), ("validation", val_ids)):
        losses = []
        for _ in range(config.eval_batches):
            x, y = get_batch(data, config, device)
            _, loss = model(x, y)
            losses.append(float(loss.item()))
        result[split] = sum(losses) / len(losses)
    model.train()
    return result


def verify_attention_and_causality(model: TinyGPT, alphabet: list[str], config: Config,
                                   device: torch.device):
    model.eval()
    batch, seq_len = 2, min(24, config.block_size)
    sample = torch.randint(0, len(alphabet), (batch, seq_len), device=device)
    logits, _, attentions = model(sample, return_attentions=True)
    weights = attentions[0]
    row_sum_error = float((weights.sum(-1) - 1.0).abs().max().item())
    upper = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
    future_mass = float(weights.masked_select(upper[None, None, :, :]).abs().max().item())
    shape_ok = tuple(logits.shape) == (batch, seq_len, len(alphabet))

    # Future-token perturbations must not affect logits at earlier positions.
    cut = seq_len // 2
    changed = sample.clone()
    changed[:, cut:] = (changed[:, cut:] + 1) % len(alphabet)
    original_logits, _ = model(sample)
    altered_logits, _ = model(changed)
    prefix_delta = float((original_logits[:, :cut] - altered_logits[:, :cut]).abs().max().item())

    # Compare the transparent implementation with PyTorch's SDPA reference API.
    torch.manual_seed(config.seed + 17)
    q = torch.randn(2, config.n_head, seq_len, config.n_embd // config.n_head, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    manual_scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    manual_scores = manual_scores.masked_fill(upper[None, None, :, :], float("-inf"))
    manual = F.softmax(manual_scores, dim=-1) @ v
    sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
    sdpa_delta = float((manual - sdpa).abs().max().item())

    return {
        "logits_shape": list(logits.shape),
        "attention_shape": list(weights.shape),
        "attention_row_sum_max_abs_error": row_sum_error,
        "future_attention_mass_max": future_mass,
        "future_token_perturbation_prefix_logits_max_abs_delta": prefix_delta,
        "manual_vs_pytorch_sdpa_max_abs_delta": sdpa_delta,
        "shape_check": bool(shape_ok),
        "causal_mask_check": bool(future_mass < 1e-7 and prefix_delta < 1e-6),
        "attention_normalization_check": bool(row_sum_error < 1e-5),
        "sdpa_equivalence_check": bool(sdpa_delta < 1e-5),
    }


def write_loss_svg(history: list[dict[str, Any]], path: Path):
    width, height = 1200, 620
    left, right, top, bottom = 100, 40, 70, 95
    plot_w, plot_h = width - left - right, height - top - bottom
    values = [r[k] for r in history for k in ("train_loss", "validation_loss")]
    y_min = max(0.0, min(values) * 0.85)
    y_max = max(values) * 1.08
    if y_max <= y_min:
        y_max = y_min + 1.0

    def px(step):
        max_step = max(history[-1]["step"], 1)
        return left + plot_w * step / max_step

    def py(value):
        return top + plot_h * (y_max - value) / (y_max - y_min)

    train_points = " ".join(f"{px(r['step']):.1f},{py(r['train_loss']):.1f}" for r in history)
    val_points = " ".join(f"{px(r['step']):.1f},{py(r['validation_loss']):.1f}" for r in history)
    grid = []
    for i in range(6):
        value = y_min + (y_max - y_min) * i / 5
        y = py(value)
        grid.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#24344a" stroke-width="1"/>')
        grid.append(f'<text x="{left-16}" y="{y+5:.1f}" text-anchor="end" fill="#9aabc2" font-size="17">{value:.2f}</text>')
    labels = []
    for i in range(6):
        step = history[-1]["step"] * i / 5
        x = px(step)
        labels.append(f'<text x="{x:.1f}" y="{height-48}" text-anchor="middle" fill="#9aabc2" font-size="16">{int(step)}</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#08111f"/>
<text x="{left}" y="36" fill="#eef6ff" font-size="24" font-weight="700">Training run · character-level TinyGPT</text>
{''.join(grid)}
{''.join(labels)}
<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#8091a8"/>
<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#8091a8"/>
<polyline points="{train_points}" fill="none" stroke="#2fd7f2" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
<polyline points="{val_points}" fill="none" stroke="#f6a35b" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
<text x="{left+8}" y="{height-12}" fill="#9aabc2" font-size="15">Optimizer step</text>
<circle cx="{width-295}" cy="35" r="6" fill="#2fd7f2"/><text x="{width-280}" y="41" fill="#d6e2f0" font-size="16">Train</text>
<circle cx="{width-195}" cy="35" r="6" fill="#f6a35b"/><text x="{width-180}" y="41" fill="#d6e2f0" font-size="16">Validation</text>
</svg>'''
    path.write_text(svg, encoding="utf-8")


def write_causal_mask_svg(path: Path, size: int = 16):
    cell, gap, margin = 28, 3, 75
    extent = margin * 2 + size * cell + (size - 1) * gap
    squares = []
    for r in range(size):
        for c in range(size):
            fill = "#24c7e8" if c <= r else "#182436"
            squares.append(f'<rect x="{margin+c*(cell+gap)}" y="{margin+r*(cell+gap)}" width="{cell}" height="{cell}" rx="3" fill="{fill}"/>')
    labels = []
    for i in range(size):
        labels.append(f'<text x="{margin+i*(cell+gap)+cell/2}" y="{margin-18}" text-anchor="middle" fill="#9fb1c8" font-size="13">{i}</text>')
        labels.append(f'<text x="{margin-18}" y="{margin+i*(cell+gap)+cell/2+5}" text-anchor="end" fill="#9fb1c8" font-size="13">{i}</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{extent}" height="{extent+35}" viewBox="0 0 {extent} {extent+35}">
<rect width="100%" height="100%" fill="#08111f"/>
<text x="{extent/2}" y="30" text-anchor="middle" font-size="20" fill="#eef6ff" font-weight="700">Causal attention mask</text>
{''.join(labels)}{''.join(squares)}
<text x="{extent/2}" y="{extent+18}" text-anchor="middle" font-size="13" fill="#9fb1c8">Columns: key positions · rows: query positions</text>
</svg>'''
    path.write_text(svg, encoding="utf-8")


def write_attention_heatmap_svg(weights: torch.Tensor, path: Path):
    """Render measured attention weights from layer 0 for a fixed validation input."""
    heads = min(4, weights.shape[0])
    seq_len = min(24, weights.shape[-1])
    cell, gap = 14, 2
    grid = seq_len * (cell + gap) - gap
    width, height = 1120, 900
    panels = []
    for head in range(heads):
        col, row = head % 2, head // 2
        x0, y0 = 70 + col * 540, 95 + row * 390
        panels.append(f'<text x="{x0}" y="{y0-22}" fill="#eef6ff" font-size="19" font-weight="700">Layer 0 · Head {head}</text>')
        panels.append(f'<rect x="{x0-8}" y="{y0-8}" width="{grid+16}" height="{grid+16}" rx="6" fill="#101e31" stroke="#32435a"/>')
        for r in range(seq_len):
            for c in range(seq_len):
                value = float(weights[head, r, c].item())
                # Dark navy at zero, bright cyan at larger attention weight.
                red = int(9 + 14 * value)
                green = int(24 + 205 * value)
                blue = int(42 + 211 * value)
                color = f"#{red:02x}{green:02x}{blue:02x}"
                panels.append(
                    f'<rect x="{x0+c*(cell+gap)}" y="{y0+r*(cell+gap)}" '
                    f'width="{cell}" height="{cell}" rx="2" fill="{color}"/>'
                )
        panels.append(f'<text x="{x0+grid/2}" y="{y0+grid+28}" text-anchor="middle" fill="#9fb1c8" font-size="13">Key position (source)</text>')
        panels.append(f'<text x="{x0-28}" y="{y0+grid/2}" transform="rotate(-90 {x0-28} {y0+grid/2})" text-anchor="middle" fill="#9fb1c8" font-size="13">Query position (target)</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#08111f"/>
<text x="{width/2}" y="38" text-anchor="middle" fill="#f0f6ff" font-size="25" font-weight="700">Measured attention weights · Layer 0</text>
<text x="{width/2}" y="66" text-anchor="middle" fill="#9fb1c8" font-size="15">Validation text · 24 character positions · brighter cells indicate larger softmax weights</text>
{''.join(panels)}
</svg>'''
    path.write_text(svg, encoding="utf-8")


def write_architecture_svg(path: Path):
    width, height = 1400, 860
    boxes = [
        (490, 55, 420, 70, "Input token IDs", "B × T"),
        (490, 150, 420, 70, "Token embedding + position embedding", "B × T × d_model"),
        (490, 250, 420, 75, "Pre-LayerNorm → Causal Multi-Head Self-Attention", "Q, K, V · causal mask"),
        (490, 365, 420, 75, "Residual connection", "x ← x + Attention(LN(x))"),
        (490, 480, 420, 75, "Pre-LayerNorm → Feed-Forward / GELU", "d_model → 4d_model → d_model"),
        (490, 595, 420, 75, "Residual connection + final LayerNorm", "contextual hidden states"),
        (490, 710, 420, 70, "Language-modeling head", "logits · B × T × vocabulary"),
    ]
    elements = []
    for x,y,w,h,title,sub in boxes:
        elements.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="#12263b" stroke="#2fd7f2" stroke-width="2"/>')
        elements.append(f'<text x="{x+w/2}" y="{y+30}" text-anchor="middle" fill="#f0f6ff" font-size="20" font-weight="700">{sax.escape(title)}</text>')
        elements.append(f'<text x="{x+w/2}" y="{y+55}" text-anchor="middle" fill="#a9bfd6" font-size="15">{sax.escape(sub)}</text>')
    arrows = []
    for i in range(len(boxes)-1):
        x=700; y1=boxes[i][1]+boxes[i][3]; y2=boxes[i+1][1]
        arrows.append(f'<line x1="{x}" y1="{y1+5}" x2="{x}" y2="{y2-7}" stroke="#8aa5bd" stroke-width="3" marker-end="url(#arrow)"/>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#8aa5bd"/></marker></defs>
<rect width="100%" height="100%" fill="#08111f"/>
<text x="700" y="34" text-anchor="middle" font-size="26" fill="#f0f6ff" font-weight="700">Decoder-only Transformer · one block shown</text>
{''.join(arrows)}{''.join(elements)}
<rect x="980" y="240" width="340" height="130" rx="14" fill="#102b3a" stroke="#f6a35b" stroke-width="2"/>
<text x="1150" y="276" text-anchor="middle" fill="#f6c38f" font-size="18" font-weight="700">Causal constraint</text>
<text x="1150" y="308" text-anchor="middle" fill="#d6e2f0" font-size="15">Queries at t can read keys</text>
<text x="1150" y="333" text-anchor="middle" fill="#d6e2f0" font-size="15">only at positions ≤ t.</text>
<line x1="910" y1="287" x2="970" y2="287" stroke="#f6a35b" stroke-width="2" stroke-dasharray="6 5"/>
<text x="700" y="832" text-anchor="middle" fill="#8fa5bd" font-size="16">The block repeats N times; diagram simplifies residual paths and tensor reshapes.</text>
</svg>'''
    path.write_text(svg, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20261114)
    args = parser.parse_args()

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    config = Config(steps=args.steps, seed=args.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = torch.device("cpu")

    train_text, val_text, alphabet, stoi, itos, train_ids, val_ids = load_corpus(DATA_PATH)
    model = TinyGPT(len(alphabet), config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    baseline_checks = verify_attention_and_causality(model, alphabet, config, device)
    initial = estimate_loss(model, train_ids, val_ids, config, device)
    history = [{"step": 0, "train_loss": initial["train"], "validation_loss": initial["validation"]}]
    rows = []

    start = time.perf_counter()
    model.train()
    for step in range(1, config.steps + 1):
        xb, yb = get_batch(train_ids, config, device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(xb, yb)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
        optimizer.step()
        rows.append({"step": step, "loss": float(loss.item()), "gradient_norm_preclip": grad_norm})
        if step % config.eval_interval == 0 or step == config.steps:
            losses = estimate_loss(model, train_ids, val_ids, config, device)
            history.append({"step": step, "train_loss": losses["train"],
                            "validation_loss": losses["validation"]})
            print(f"step={step:04d}/{config.steps} train={losses['train']:.4f} "
                  f"validation={losses['validation']:.4f} grad_norm={grad_norm:.4f}")
    elapsed = time.perf_counter() - start
    final = estimate_loss(model, train_ids, val_ids, config, device)

    model.eval()
    prompt = "El Transformer"
    prompt_ids = torch.tensor([[stoi.get(c, 0) for c in prompt]], dtype=torch.long, device=device)
    generated_ids = model.generate(prompt_ids, max_new_tokens=280, temperature=0.85, top_k=24)
    generated_text = "".join(itos[int(i)] for i in generated_ids[0].tolist())
    final_checks = verify_attention_and_causality(model, alphabet, config, device)

    with (EVIDENCE / "training-log.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["step", "loss", "gradient_norm_preclip"])
        writer.writeheader()
        writer.writerows(rows)
    (EVIDENCE / "loss-history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (EVIDENCE / "generated-sample.txt").write_text(generated_text + "\n", encoding="utf-8")
    write_loss_svg(history, EVIDENCE / "training-validation-loss.svg")
    write_causal_mask_svg(EVIDENCE / "causal-mask.svg")
    probe_text = val_text[:min(config.block_size, 24)]
    probe_ids = torch.tensor([[stoi[c] for c in probe_text]], dtype=torch.long, device=device)
    _, _, probe_attentions = model(probe_ids, return_attentions=True)
    write_attention_heatmap_svg(probe_attentions[0][0], EVIDENCE / "attention-heads-heatmap.svg")
    write_architecture_svg(EVIDENCE / "decoder-only-architecture.svg")
    torch.save({"model_state_dict": model.state_dict(), "config": asdict(config),
                "itos": itos, "stoi": stoi}, MODELS / "tiny-gpt-es.pt")

    metrics = {
        "lab": "Consola Abierta - Transformer Internals 01",
        "run_timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "xpu_available": bool(getattr(torch, "xpu", None) and torch.xpu.is_available()),
        "cpu_threads": torch.get_num_threads(),
        "cpu": "11th Gen Intel Core i5-1155G7 @ 2.50 GHz · 4 cores / 8 logical processors",
        "gpu_detected": "Intel Iris Xe Graphics (not used for this PyTorch run)",
        "dataset": {
            "train_characters": len(train_text),
            "validation_characters": len(val_text),
            "vocabulary_characters": len(alphabet),
            "tokenization": "character-level, pedagogical only",
            "train_validation_split": "separate authored sections; validation never used for optimizer updates"
        },
        "model": asdict(config) | {"vocab_size": len(alphabet), "trainable_parameters": n_params,
                                   "architecture": "decoder-only, Pre-LayerNorm, learned positions, tied token/output embeddings"},
        "training": {
            "steps_completed": config.steps,
            "wall_time_seconds": round(elapsed, 3),
            "initial_train_loss": initial["train"],
            "final_train_loss": final["train"],
            "initial_validation_loss": initial["validation"],
            "final_validation_loss": final["validation"],
            "train_loss_reduction_percent": round((initial["train"] - final["train"]) / initial["train"] * 100, 2),
            "validation_loss_reduction_percent": round((initial["validation"] - final["validation"]) / initial["validation"] * 100, 2)
        },
        "checks": {"before_training": baseline_checks, "after_training": final_checks},
        "interpretation": [
            "This is a small character-level teaching model, not a production language model.",
            "Loss values depend on this corpus, seed, configuration and hardware; they are not comparable to external model benchmarks.",
            "CPU-only execution was intentional because this host has no CUDA or XPU device available."
        ]
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    report = (
        "# Lab report: TinyGPT decoder-only Transformer\n\n"
        f"- Host: {metrics['host']}\n- Python: {metrics['python']}\n"
        f"- PyTorch: {metrics['pytorch']}\n- Device: CPU\n"
        f"- Trainable parameters: {n_params:,}\n- Vocabulary: {len(alphabet)} characters\n"
        f"- Train/validation characters: {len(train_text):,} / {len(val_text):,}\n"
        f"- Steps: {config.steps:,}\n- Training wall time: {elapsed:.3f} s\n"
        f"- Train loss: {initial['train']:.4f} → {final['train']:.4f}\n"
        f"- Validation loss: {initial['validation']:.4f} → {final['validation']:.4f}\n\n"
        "## Automated checks\n\n"
        + "\n".join(f"- {key}: {value}" for key, value in final_checks.items())
        + "\n\n## Limitations\n\n"
        + "\n".join(f"- {item}" for item in metrics["interpretation"])
        + "\n"
    )
    (REPORTS / "lab-report.md").write_text(report, encoding="utf-8")

    print("\n=== RUN SUMMARY ===")
    print(f"parameters={n_params:,} vocab={len(alphabet)} device={device}")
    print(f"train_loss={initial['train']:.4f} -> {final['train']:.4f}")
    print(f"validation_loss={initial['validation']:.4f} -> {final['validation']:.4f}")
    print(f"wall_time_seconds={elapsed:.3f}")
    print("checks=" + json.dumps(final_checks, ensure_ascii=False))
    print("\n=== GENERATED SAMPLE ===")
    print(generated_text)
    failed = [k for k, v in final_checks.items() if k.endswith("_check") and not v]
    if failed:
        print("FAILED CHECKS:", ", ".join(failed))
        return 2
    if final["train"] >= initial["train"]:
        print("WARNING: training loss did not decrease in this run.")
        return 3
    print("\nLAB STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
