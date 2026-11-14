# Lab report: Self-Attention deep dive

- Host: PCRAY
- OS: Windows-11-10.0.26300-SP0
- Python: 3.12.10
- PyTorch: 2.14.0+cpu
- Device: CPU
- CPU threads: 4
- Runtime: 0.0537 s

## Gradient validation

- Manual Q/K/V gradients vs. autograd max absolute error: 3.469e-18
- W_Q[0,0] finite difference vs. autograd absolute error: 7.679e-11
- Gradient L2 norms: {"input_x": 0.14022487511256485, "W_Q": 0.17777623787452146, "W_K": 0.18282424220858304, "W_V": 0.5377044261880152, "W_O": 0.5280069815081879}

## Scaling experiment

| d_k | Entropy unscaled | Entropy scaled | Max weight unscaled | Max weight scaled |
|---:|---:|---:|---:|---:|
| 8 | 1.4913 | 2.7473 | 0.5322 | 0.1934 |
| 16 | 1.0284 | 2.7388 | 0.6519 | 0.1948 |
| 32 | 0.6919 | 2.7504 | 0.7567 | 0.1904 |
| 64 | 0.4659 | 2.7463 | 0.8248 | 0.1922 |
| 128 | 0.3301 | 2.7326 | 0.8717 | 0.1938 |
| 256 | 0.1967 | 2.7390 | 0.9238 | 0.1943 |

## GQA / SDPA tensor check

- Query shape: [2, 8, 12, 8]
- KV shape before sharing expansion: [2, 2, 12, 8]
- Expanded KV shape: [2, 8, 12, 8]
- Output shape: [2, 8, 12, 8]
- Manual attention vs. PyTorch SDPA max abs delta: 3.576e-07

## RoPE identity

- Relative-position rotation identity max absolute error: 0.000e+00

## Automated checks

- manual_gradient_matches_autograd: PASS
- finite_difference_matches_autograd: PASS
- causal_mask_blocks_future_positions: PASS
- attention_rows_sum_to_one: PASS
- manual_attention_matches_pytorch_sdpa: PASS
- gqa_kv_expansion_equivalence: PASS
- rope_relative_position_identity: PASS

## Limitations

- Synthetic random tensors isolate mathematical properties; they are not model-quality or language-understanding measurements.
- The scaling experiment uses a fixed seed and standard-normal Q/K vectors; trained representations may have different statistics.
- Parameter and KV-cache tables are analytical estimates for one layer, one sequence and BF16 cache storage, not measured GPU memory.
- The CPU lab explicitly expands shared KV heads for GQA; it does not benchmark the native CUDA enable_gqa path.
- The RoPE test checks the relative-position rotation identity, not language-model quality or long-context extrapolation.
- Attention weights are internal routing coefficients, not a complete or automatically faithful explanation of model decisions.
