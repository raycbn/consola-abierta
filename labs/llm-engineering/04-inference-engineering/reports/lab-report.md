# Lab report — Prefill, Decode and KV Cache

- Host: PCRAY
- Python: 3.12.10
- PyTorch: 2.14.0+cpu
- Device: CPU · float32 · 4 configured CPU threads
- Model: 3 decoder blocks · 4 heads · d_model 64
- Batch: 2
- Prompt: 24 token positions
- Incremental decode inputs: 12
- Seed: 20261114

## Numerical equivalence

- Prefill vs full-forward max absolute logit error: 0
- Maximum error across incremental decode: 1.14440917969e-05
- Comparisons: one prefill plus 12 incremental steps.
- Each incremental output is compared with the last-position logits from a full-prefix forward pass.

## KV cache accounting

- Final sequence length: 36
- Measured K/V tensor storage: 110,592 bytes
- Formula result: 110,592 bytes
- Cached-path model input positions: 36
- Recomputed full-prefix model input positions: 366
- Structural reduction in input positions: 90.16%

## Illustrative production-sized memory calculation

For 32 layers, batch 1, 8 KV heads, head_dim 128 and 2 bytes per value:
- 16,384 cached tokens: 2,147,483,648 bytes (2.00 GiB).
- Formula: 2 × layers × batch × tokens × KV heads × head_dim × bytes_per_element.

## Automated checks

- prefill_logits_match_full_forward: PASS
- all_decode_logits_match_full_recompute: PASS
- cache_length_increments_once_per_token: PASS
- all_layers_have_expected_cache_shape: PASS
- measured_cache_bytes_match_formula: PASS
- cached_path_processes_fewer_token_positions: PASS
- logit_errors_are_finite: PASS

**LAB STATUS: PASS**

## Boundaries

- The model is randomly initialised and validates inference mechanics only. It does not generate meaningful language and is not a trained LLM.
- This CPU run validates numerical equivalence and cache accounting. It makes no GPU acceleration, throughput or latency claim.
- The structural count does not equal FLOPs: the new query still reads the accumulated K/V context at each decode step.
- The memory example assumes full attention with the stated KV head count and dtype. It excludes allocator fragmentation, metadata, workspaces, weights and activations.
- No model weights, CUDA runtime, external dataset, hosted endpoint, API or paid service is downloaded or required.
