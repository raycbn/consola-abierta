# Lab report — How an LLM Learns

- Host: PCRAY
- Python: 3.12.10
- PyTorch: 2.14.0+cpu
- Device: CPU (4 threads)
- Vocabulary: 66 characters
- Train / validation characters: 7,571 / 2,413
- Parameters: 62,880
- Steps per run: 240

## Controlled comparison

| Policy | Train loss | Validation loss | Seconds |
|---|---:|---:|---:|
| constant | 4.1908 → 2.3869 | 4.1914 → 2.4630 | 8.994 |
| warmup_cosine | 4.1908 → 2.5902 | 4.1914 → 2.6363 | 7.712 |

## Numerical checks

- shift_alignment_check: True
- cross_entropy_vs_log_softmax_nll_max_abs_error: 0.0
- cross_entropy_equivalence_check: True
- cross_entropy_gradient_max_abs_error: 1.3877787807814457e-17
- cross_entropy_gradient_check: True
- adamw_zero_gradient_expected_parameter: 0.98
- adamw_zero_gradient_actual_parameter: 0.9800000190734863
- adamw_decoupled_decay_check: True
- gradient_norm_before_clip: 5.0
- gradient_norm_after_clip: 0.9999998211860657
- gradient_clipping_check: True
- warmup_start_positive_check: True
- warmup_peak_reached_check: True
- cosine_minimum_reached_check: True
- cosine_decay_monotonic_check: True
- both_policies_reduce_train_loss_check: True
- both_policies_reduce_validation_loss_check: True
- all_final_losses_finite_check: True

## Boundaries

- The two runs start from identical weights and consume the same precomputed batches.
- This controlled comparison isolates the learning-rate policy within this small experiment; it does not prove one schedule is universally superior.
- The corpus is a small authored character-level teaching corpus, not a representative pretraining dataset.
- Validation loss is measured but never used for gradient updates.
- No GPU, API, hosted model, external dataset or paid service is used.
