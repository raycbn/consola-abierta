# LLM Engineering 06 — Quantization fundamentals

Laboratorio didáctico reproducible sobre precisión numérica y cuantización de pesos y activaciones.

## Requisitos

- Python 3.10 o superior
- Solo biblioteca estándar de Python
- Sin pip install, PyTorch, Transformers, modelos descargados, GPU ni CUDA
- Coste: 0 €

## Ejecución

En Windows PowerShell:

    .\run.ps1

También se puede ejecutar directamente:

    python .\src\run_lab.py

El laboratorio fija una semilla (20261114) y genera las evidencias y los informes desde cero.

## Experimentos

1. FP16 (IEEE binary16) frente a BF16: resolución, rango dinámico, overflow y underflow.
2. RTN (Round-to-Nearest) simétrico INT8 e INT4.
3. Cuantización afín asimétrica con zero-point.
4. Escalas per-tensor, per-output-channel y per-group (16 y 32 elementos).
5. Error de reconstrucción: MAE, MSE, RMSE, máximo error, SQNR y relative L2.
6. Drift en una proyección lineal sintética W4A16-style y W8A8-style.
7. Calibración de activaciones: absmax frente a clipping del percentil 99.
8. Transformación algebraica de canales W' = W D y x' = D^-1 x.
9. Aritmética ideal de memoria para un modelo de 7B parámetros.
10. Estimación ideal de KV Cache para una configuración GQA representativa.

## Evidencias

- evidence/weight-quantization-metrics.csv
- evidence/linear-output-drift.csv
- evidence/activation-calibration.csv
- evidence/float-format-rounding.csv
- evidence/model-memory-estimates.csv
- evidence/kv-cache-estimates.csv
- reports/metrics.json
- reports/lab-report.md

## Resultados reproducidos en el entorno de publicación

- Python 3.12.10
- 12/12 comprobaciones PASS
- Matriz sintética: 32 × 64
- 24 vectores de activación
- INT4 per-tensor MSE: 0.157819
- INT4 per-channel MSE: 0.028064
- INT4 per-group-16 MSE: 0.014677
- W4A16-style output RMSE: 0.848397
- W8A8-style output RMSE: 0.146636
- FP16 overflows at 100,000; BF16 remains finite (99,840)
- Percentile 99 clipping: 16 / 1,536 values clipped; the RMSE is worse in this deliberately outlier-heavy sample.

These values are evidence for this deterministic synthetic workload only.

## Boundaries — read before interpreting

This is a numerical teaching simulator, not a model quantizer and not an inference runtime.

- The only integer quantization algorithm implemented is a linear RTN baseline.
- GPTQ, AWQ and SmoothQuant are explained in the article; they are not reproduced by this code.
- There are no FP8 kernels or true FP8 GEMMs.
- The W4A16/W8A8 labels describe precision paths in a toy matrix-vector product; they are not vendor kernel implementations.
- The memory tables are ideal bit-count arithmetic. They exclude runtime allocations, unquantized layers, packing/alignment and most metadata.
- No perplexity, benchmark task, generation quality, latency, throughput, VRAM allocation, or tokens/s is measured.
- Lower reconstruction MSE does not guarantee better language-model quality.

For deployment decisions, use representative calibration data, a supported checkpoint format and backend, task-level evaluation, generation regression tests and real workload benchmarks.
