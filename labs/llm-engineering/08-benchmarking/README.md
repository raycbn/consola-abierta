# LLM Engineering 08 — LLM benchmarking and trace analysis

A reproducible, standard-library-only lab for analysing request-level and streamed-token traces from LLM inference servers.

## Cost and requirements

- Python 3.10+
- Python standard library only
- No PyTorch, Transformers, vLLM, CUDA, GPU, external API or model download required
- Cost: €0

## Quick start

Windows PowerShell:

~~~powershell
.\run.ps1
~~~

Or:

~~~powershell
python .\src\run_lab.py
~~~

The default run creates a deterministic teaching fixture (seed 20261114) with two traffic profiles (steady and bursty), then generates:

- data/requests.csv
- data/token-events.csv
- data/manifest.json
- reports/request-metrics.csv
- reports/itl-samples.csv
- reports/metrics.json
- reports/lab-report.md

The 48-request-per-scenario sample is intentionally small and synthetic. Its latencies and throughput are not results from any LLM, GPU, API or serving engine.

## Analyse your own trace

The analyser accepts request-level client observations plus an optional file of streamed output events:

~~~powershell
python .\src\run_lab.py --requests C:\bench\requests.csv --events C:\bench\token-events.csv --output .\reports-real --slo-ttft-ms 500 --slo-tpot-ms 80 --slo-e2e-ms 5000
~~~

Optional cost model (all prices are supplied by the operator):

~~~powershell
python .\src\run_lab.py --requests C:\bench\requests.csv --events C:\bench\token-events.csv --gpu-hour-cost 2.40 --gpu-count 4
~~~

This is a simple direct-compute estimate over the trace's measurement window. It excludes idle capacity, hardware amortisation, electricity, networking, storage, licences, platform services, orchestration and operations. It is not a cloud price catalogue.

### Request CSV schema

Required columns:

| Column | Meaning |
|---|---|
| request_id | Unique request identifier |
| scenario | Workload/configuration name, e.g. baseline, long-context, bursty |
| scheduled_ms | Planned arrival timestamp on the benchmark client's monotonic clock |
| sent_ms | Actual time the client sent the request |
| first_token_ms | First non-empty streamed content timestamp; blank if no content arrived |
| completed_ms | Final response/completion timestamp |
| prompt_tokens | Input token count for this request |
| output_tokens | Generated output token count |
| status | success, error, timeout, rejected or cancelled |

Optional columns:

- server_queue_ms: server-side scheduler queue time, only when measured by the serving runtime.
- server_prefill_ms: server-side prefill time, when available.
- server_decode_ms: server-side decode time, when available.

All timestamps must share one monotonic clock domain and use milliseconds. Do not mix wall-clock timestamps, client clocks and server clocks without clock synchronisation and a documented correction. The analyser calculates client queue as sent_ms - scheduled_ms and client TTFT as first_token_ms - sent_ms.

### Stream event CSV schema

Required columns:

| Column | Meaning |
|---|---|
| request_id | Must match a request |
| chunk_index | 1-based, contiguous sequence of non-empty streamed output events |
| received_ms | Client-observed receipt timestamp for the event |
| tokens_in_chunk | Number of generated tokens represented by this event |

When one streaming event contains multiple tokens, ITL counts the time between events, not an artificial zero-time gap between tokens inside the same event. The request-level TPOT calculation amortises the decode window over generated tokens. This distinction matters for speculative decoding and server-side stream coalescing.

If token events are unavailable, request-level TTFT, TPOT and E2E remain analysable, but ITL is not derived. Do not manufacture token timestamps from the total request duration.

## Metric definitions used by this lab

- **TTFT:** first non-empty streamed content receipt minus actual client send time.
- **ITL:** time between consecutive non-empty streamed content events. This lab pools the event gaps.
- **TPOT:** (E2E - TTFT) / (output_tokens - 1), calculated once per request when output length is greater than one.
- **E2E:** final response completion minus actual client send time.
- **Client queue:** actual send timestamp minus planned schedule timestamp. This is load-generator waiting, not server queue time.
- **Output token throughput:** successful generated tokens divided by measurement-window seconds.
- **Total token throughput:** input plus output tokens divided by measurement-window seconds.
- **Request throughput:** successful requests divided by measurement-window seconds.
- **Goodput:** successful requests within the configured TTFT, TPOT (where defined) and E2E thresholds divided by measurement-window seconds.

Percentiles use linear interpolation (R-7). A percentile is not a confidence interval. The report includes p50, p90, p95 and p99, but a small sample cannot support a robust production tail-latency SLO.

The measurement window starts at the earliest scheduled request and ends at the latest completion. It includes drain time and does not include warmup. Failed requests remain in request/error counts but do not contribute to successful token throughput.

## Evidence and verification

The analyser runs 20 consistency checks covering timestamps, request identity, streamed-event order, token reconciliation, latency definitions and throughput/goodput arithmetic. All generated files are reproducible from the fixed seed.

## Boundaries

This lab analyses timestamped observations. It is not a load generator, a model runtime or a GPU benchmark. It cannot establish:

- the performance of any model or accelerator;
- kernel-level execution time;
- an engine's actual batch formation or scheduler decisions;
- network latency decomposition without synchronised client/server spans;
- confidence intervals for a production percentile from a small fixture;
- the quality or correctness of generated answers.

For a real deployment, collect client-side request traces and, where possible, server-side runtime metrics. Keep them as distinct measurement planes. Do not add client queue, server queue, TTFT and prefill values together unless the exact instrumentation boundaries prove that they are disjoint.

## Suggested next experiment

Capture the same workload at several offered request rates. Preserve the prompt and output length distributions, compare p50/p95/p99 latency with aggregate token throughput and SLO goodput, and identify the knee where throughput gains stop compensating for tail-latency growth.
