# Lab report: TinyGPT decoder-only Transformer

- Host: PCRAY
- Python: 3.12.10
- PyTorch: 2.14.0+cpu
- Device: CPU
- Trainable parameters: 110,464
- Vocabulary: 66 characters
- Train/validation characters: 7,571 / 2,413
- Steps: 600
- Training wall time: 53.383 s
- Train loss: 4.2293 → 2.1660
- Validation loss: 4.2222 → 2.3091

## Automated checks

- logits_shape: [2, 24, 66]
- attention_shape: [2, 4, 24, 24]
- attention_row_sum_max_abs_error: 2.384185791015625e-07
- future_attention_mass_max: 0.0
- future_token_perturbation_prefix_logits_max_abs_delta: 0.0
- manual_vs_pytorch_sdpa_max_abs_delta: 2.980232238769531e-07
- shape_check: True
- causal_mask_check: True
- attention_normalization_check: True
- sdpa_equivalence_check: True

## Limitations

- This is a small character-level teaching model, not a production language model.
- Loss values depend on this corpus, seed, configuration and hardware; they are not comparable to external model benchmarks.
- CPU-only execution was intentional because this host has no CUDA or XPU device available.
