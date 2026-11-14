# Lab report — LLM quantization fundamentals

## Reproducibility
- Date: 2026-11-14
- Python: 3.12.10
- Dependencies: standard library only; no pip install required
- Seed: 20261114
- Checks: 12/12 PASS
- Execution: CPU numerical emulation; no GPU or CUDA

## What was tested
- Synthetic weight matrix: 32 output channels × 64 input features
- Synthetic activation vectors: 24
- Linear Round-to-Nearest baseline: symmetric INT8/INT4 and asymmetric affine INT4
- Scale granularities: per-tensor, per-output-channel and per-group (16/32 values)
- Floating-point rounding: IEEE binary16 and bfloat16
- Activation calibration: absmax vs 99th-percentile clipping
- Linear projection drift: W4A16-style and W8A8-style educational paths
- Algebraic channel rescaling: W' = W D and x' = D^-1 x

## Weight reconstruction results

| Scheme | MSE | RMSE | Max abs error | SQNR (dB) | Effective bits/weight estimate |
|---|---:|---:|---:|---:|---:|
| FP16 emulation | 8.27e-08 | 0.0002876186 | 0.003125 | 73.2042 | 16.0000 |
| BF16 emulation | 4.7354e-06 | 0.0021760961 | 0.014466906 | 55.6270 | 16.0000 |
| INT8 symmetric per-tensor | 0.0005355575 | 0.023142115 | 0.04133163 | 35.0925 | 8.0156 |
| INT8 symmetric per-channel | 8.56259e-05 | 0.0092534286 | 0.041298152 | 43.0545 | 8.5000 |
| INT4 symmetric per-tensor | 0.15781924 | 0.39726469 | 0.74925456 | 10.3989 | 4.0156 |
| INT4 symmetric per-channel | 0.02806436 | 0.16752421 | 0.74564722 | 17.8990 | 4.5000 |
| INT4 symmetric per-group-16 | 0.014676869 | 0.12114813 | 0.72916069 | 20.7142 | 6.0000 |
| INT4 symmetric per-group-32 | 0.019988561 | 0.14138091 | 0.72916069 | 19.3727 | 5.0000 |
| INT4 asymmetric per-tensor | 0.12335874 | 0.35122463 | 0.65596377 | 11.4688 | 4.0156 |

Effective bits include nominal integer payload plus one FP32 scale per quantization region. Zero points, packing alignment, other metadata, unquantized tensors and runtime buffers are excluded.

## Linear projection drift

| Path | RMSE | Max abs error | Relative L2 |
|---|---:|---:|---:|
| FP32 reference | 0 | 0 | 0 |
| W4A16 educational: INT4 group-16 weights, FP32 activations | 0.84839699 | 5.2115368 | 0.096979814 |
| W8A8 educational: INT8 per-channel weights, INT8 per-vector activations | 0.14663564 | 1.200875 | 0.016761843 |

Activations remain FP32 in the W4A16-style path. The W8A8-style path quantizes each input vector independently to symmetric INT8. These labels explain a precision policy; they do not reproduce a vendor kernel or a checkpoint format.

## Activation calibration

| Strategy | Threshold | Scale | Clipped values | RMSE | Max abs error |
|---|---:|---:|---:|---:|---:|
| absmax | 13 | 0.1023622 | 0 | 0.029263693 | 0.051141176 |
| percentile_99 | 1.8431123 | 0.014512695 | 16 | 0.49341934 | 11.156888 |

Clipping trades saturation for finer resolution over the retained range. A production threshold must be selected using representative calibration data and downstream validation.

## Floating-point format probes

| Input | FP16 emulation | FP16 finite | BF16 emulation | BF16 finite |
|---:|---:|:---:|---:|:---:|
| 0.1 | 0.0999755859375 | True | 0.10009765625 | True |
| 0.3333333 | 0.333251953125 | True | 0.333984375 | True |
| 1 | 1.0 | True | 1.0 | True |
| -2.71828 | -2.71875 | True | -2.71875 | True |
| 65504 | 65504.0 | True | 65536.0 | True |
| 70000 | inf | False | 70144.0 | True |
| 1e-05 | 1.0013580322265625e-05 | True | 1.0013580322265625e-05 | True |
| 100000 | inf | False | 99840.0 | True |
| -100000 | -inf | False | -99840.0 | True |
| 1e-08 | 0.0 | True | 1.0011717677116394e-08 | True |

FP16 has more significand precision than BF16 at a similar bit width; BF16 has a much wider exponent range. This is a format conversion emulation, not a GPU throughput test.

## Model weight memory arithmetic

| Representation | Bits/parameter assumption | 7B parameters (GB decimal) | 7B parameters (GiB binary) |
|---|---:|---:|---:|
| FP32 | 32 | 28.0000 | 26.0770 |
| FP16 | 16 | 14.0000 | 13.0385 |
| BF16 | 16 | 14.0000 | 13.0385 |
| FP8 | 8 | 7.0000 | 6.5193 |
| INT8 | 8 | 7.0000 | 6.5193 |
| INT4 raw payload | 4 | 3.5000 | 3.2596 |
| INT4 + FP32 scale per 64 weights | 4.5 | 3.9375 | 3.6671 |

Ideal payload arithmetic. It is not a GPU memory measurement. Only the explicitly labelled INT4+scale row includes a scale overhead estimate.

## KV cache arithmetic

| Representation | Ideal payload (GiB) |
|---|---:|
| FP16/BF16 payload | 16.0000 |
| FP8 payload | 8.0000 |
| INT8 payload | 8.0000 |
| 4-bit payload | 4.0000 |

Scenario: 32 layers, batch 16, 8192 tokens/sequence, 8 KV heads, head dimension 128.
Formula: 2 × layers × batch × tokens × kv_heads × head_dim × bytes_per_element. Scale metadata and implementation buffers are excluded.

## Checks

- [x] **PASS** — Per-channel INT4 beats per-tensor INT4 on heterogeneous rows: per-channel=0.0280643605; per-tensor=0.1578192359
- [x] **PASS** — Groupwise INT4 improves on per-tensor INT4 for this matrix: group-16=0.0146768685; per-tensor=0.1578192359
- [x] **PASS** — INT8 has lower weight reconstruction MSE than INT4: Same synthetic matrix and per-channel granularity.
- [x] **PASS** — BF16 preserves a large finite value that overflows FP16: FP16(100000)=inf; BF16(100000)=99840.0
- [x] **PASS** — Affine quantization represents real zero through zero-point: Asymmetric affine INT4 maps real zero exactly.
- [x] **PASS** — INT4 q values remain inside the configured range: Symmetric INT4 uses the explicit educational range [-7, 7].
- [x] **PASS** — Algebraic channel rescaling preserves linear output: maximum absolute output drift=8.881784197001252e-15
- [x] **PASS** — W4A16 and W8A8 paths produce finite output errors: Synthetic numerical paths only; no hardware kernel.
- [x] **PASS** — Memory arithmetic halves nominal payload from FP16 to FP8: 7B parameters: FP16=14 GB decimal; FP8=7 GB decimal, before overhead.
- [x] **PASS** — KV payload arithmetic halves from 16-bit to 8-bit: GQA example; scale metadata excluded.
- [x] **PASS** — Percentile activation calibration reports clipped values: Clipped 16 of 1536.
- [x] **PASS** — All six evidence tables contain rows: Six deterministic CSV tables generated.

## Interpretation and limits

- Inputs are deterministic synthetic values, not pretrained weights or real activations.
- Only an educational linear RTN quantizer is implemented. GPTQ, AWQ and SmoothQuant are studied in the article, not reimplemented in this lab.
- Per-channel/per-group rankings depend on the generated tensor distribution and are not universal.
- Reconstruction error and SQNR do not replace perplexity, task evaluation, generation regression tests or human review.
- The lab does not measure latency, throughput, tokens/s, GPU memory, model quality or hardware acceleration.
- FP8 is discussed in the article; no FP8 kernel or GEMM implementation is claimed.

## Evidence files

- evidence/weight-quantization-metrics.csv
- evidence/linear-output-drift.csv
- evidence/activation-calibration.csv
- evidence/float-format-rounding.csv
- evidence/model-memory-estimates.csv
- evidence/kv-cache-estimates.csv
- reports/metrics.json
