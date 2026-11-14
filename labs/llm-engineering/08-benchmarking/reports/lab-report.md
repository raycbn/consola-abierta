# LLM benchmarking trace report

 > Synthetic teaching trace. No model, API, GPU or serving engine was executed.

## Overall
- Requests: 96 (92 successful, 4 failed)
- Measurement window: 32.953 s
- Request throughput: 2.792 req/s
- Output token throughput: 70.645 tok/s
- Total token throughput: 2694.475 tok/s
- Maximum in-flight concurrency: 11
- SLO goodput: 2.701 req/s (89 requests)

## Latency distributions (ms)

| Metric | n | Mean | p50 | p90 | p95 | p99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| TTFT | 92 | 264.37 | 259.61 | 464.81 | 467.44 | 530.12 | 550.72 |
| TPOT | 92 | 20.95 | 19.81 | 29.07 | 31.03 | 36.30 | 39.83 |
| ITL | 1652 | 28.24 | 26.34 | 38.21 | 42.18 | 45.98 | 46.93 |
| E2E | 92 | 771.49 | 713.28 | 1205.94 | 1343.83 | 1659.91 | 1863.36 |
| CLIENT_QUEUE | 96 | 0.44 | 0.00 | 0.00 | 7.00 | 7.00 | 7.00 |
| SERVER_QUEUE | 92 | 13.83 | 0.00 | 69.60 | 82.80 | 96.00 | 96.00 |

## Scenario comparison

| Scenario | Requests | RPS | Output tok/s | TTFT p95 ms | TPOT p95 ms | E2E p95 ms | Max concurrency |
|---|---:|---:|---:|---:|---:|---:|---:|
| bursty | 48 | 1.396 | 35.565 | 504.17 | 34.18 | 1349.67 | 8 |
| steady | 48 | 1.776 | 44.633 | 467.32 | 22.00 | 1204.17 | 3 |

## Interpretation limits

- Percentiles use linear interpolation (R-7) over the supplied samples.
- p95/p99 are descriptive; 48 requests per scenario are not enough to establish a production tail-latency SLO.
- ITL is measured between streamed content events, not between internal model tokens when a chunk bundles several tokens.
- TPOT is calculated once per successful request as (E2E − TTFT) / (output tokens − 1).
- Client queue time is scheduled_ms to sent_ms. Server queue time is independent and must not be added blindly to client latency.
- Throughput uses earliest scheduled arrival to latest completion, includes drain time and excludes warmup (not present in this fixture).
- Synthetic latencies are validation fixtures, not model, GPU, network or engine benchmarks.
