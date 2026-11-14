# Self-Attention Deep Dive — Lab 02

A CPU-only, reproducible laboratory for the second LLM Engineering investigation.

## What it verifies

1. Scaled vs. unscaled dot-product attention: softmax entropy and average maximum weight as the head dimension grows.
2. Q/K/V reverse-mode derivatives: the manually derived gradients are compared against PyTorch autograd.
3. A central finite-difference check validates one learned query-projection weight against autograd.
4. A causal mask blocks future positions and attention rows remain normalised.
5. The manual attention implementation is compared with PyTorch scaled_dot_product_attention.
6. Grouped Query Attention is represented by sharing K/V heads across query-head groups; explicit KV expansion is checked against SDPA.
7. A RoPE relative-position rotation identity is checked numerically.
8. Analytical tables estimate attention projection parameters and the KV-cache footprint for MHA, GQA and MQA.

## Requirements

- Python 3.12 recommended.
- PyTorch 2.14.0.
- Windows, macOS or Linux.
- CPU is sufficient. No model checkpoints, external datasets, GPU, cloud service or API key are required.

Install the CPU wheel:

    python -m pip install -r requirements.txt

Run:

    .\run.ps1

Or:

    python -u .\src\run_lab.py

## Output

- reports/metrics.json: machine-readable metrics and check results.
- reports/lab-report.md: concise experiment report.
- evidence/scaling-experiment.csv: raw entropy and maximum-weight observations.
- evidence/attention-variants.csv: projection-parameter estimates.
- evidence/kv-cache-estimates.csv: one-layer BF16 KV-cache estimates.
- evidence/visuals/: generated SVG diagrams and chart.

## Reproducibility

Fixed seed: 20261115. The script records Python, PyTorch, host, OS, CPU thread count and runtime. Double precision is used for gradient validation. The scaling experiment uses 512 synthetic query vectors per head dimension and 24 candidate keys per query.

## Important boundaries

This lab isolates mathematical properties with synthetic tensors. It does not train a language model and does not measure model quality, GPU throughput, latency or actual GPU memory allocation. The GQA CPU test explicitly repeats shared KV heads for the computation; it does not benchmark PyTorch's native CUDA GQA kernels. Cache figures are analytical estimates for one layer, one sequence, BF16 storage and the stated dimensions. RoPE validation checks an algebraic identity, not long-context extrapolation.

## Primary references

- Vaswani et al. (2017), Attention Is All You Need: https://arxiv.org/abs/1706.03762
- Ainslie et al. (2023), GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints: https://arxiv.org/abs/2305.13245
- Su et al. (2021), RoFormer: Enhanced Transformer with Rotary Position Embedding: https://arxiv.org/abs/2104.09864
- PyTorch scaled_dot_product_attention: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- PyTorch Autograd mechanics: https://docs.pytorch.org/docs/stable/notes/autograd.html

Cost: €0. No paid services or APIs are used.
