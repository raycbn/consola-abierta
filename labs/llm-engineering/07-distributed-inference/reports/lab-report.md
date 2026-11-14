# Lab report — Distributed LLM inference concepts

## Reproducibility
- Date: 2026-11-14
- Python: 3.12.10
- Dependencies: Python standard library only
- Seed: 20261114
- Checks: 16/16 PASS
- Execution: single-process CPU simulation; no GPU or distributed processes

## Model arithmetic
- Hidden size: 4096
- FFN intermediate size: 11008
- Transformer layers: 32
- Active sequences: 16
- Toy layer parameter count (weights only): 6,476,005,376
- Ideal FP16/BF16 weight payload: 12.0625 GiB
- Excluded: embeddings, LM head, norms, biases, model-specific tensors, padding, runtime buffers and KV Cache

## Tensor Parallelism — communication and memory

| TP ranks | Ideal weight GiB/rank | Ring send MiB/rank/decode iteration | Fabric A ms (illustrative) | Fabric B ms (illustrative) |
|---:|---:|---:|---:|---:|
| 1 | 12.0625 | 0.0000 | 0.000000 | 0.000000 |
| 2 | 6.0312 | 8.0000 | 0.553943 | 1.447772 |
| 4 | 3.0156 | 12.0000 | 1.598915 | 4.091658 |
| 8 | 1.5078 | 14.0000 | 3.657400 | 9.253601 |

The communication model assumes two row-parallel reductions per Transformer layer and a ring AllReduce. It computes 2(p−1)/p × N × bytes sent per rank per collective. Fabric A/B use hypothetical alpha and bandwidth values to demonstrate sensitivity. Times are not GPU/NCCL benchmarks and omit overlap, topology routing, protocol, kernel scheduling and contention.

## Pipeline Parallelism — fill/drain schedule

| PP stages | Microbatches | First result (work units) | Makespan (work units) | Aggregate stage utilization | Ideal balanced fill/drain reference |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 34.5387 | 34.5387 | 1.0000 | 1.0000 |
| 1 | 2 | 34.5387 | 69.0773 | 1.0000 | 1.0000 |
| 1 | 4 | 34.5387 | 138.1546 | 1.0000 | 1.0000 |
| 1 | 8 | 34.5387 | 276.3093 | 1.0000 | 1.0000 |
| 1 | 16 | 34.5387 | 552.6186 | 1.0000 | 1.0000 |
| 1 | 32 | 34.5387 | 1105.2372 | 1.0000 | 1.0000 |
| 2 | 1 | 32.9422 | 32.9422 | 0.5000 | 0.5000 |
| 2 | 2 | 32.9422 | 50.2116 | 0.6561 | 0.6667 |
| 2 | 4 | 32.9422 | 84.7502 | 0.7774 | 0.8000 |
| 2 | 8 | 32.9422 | 153.8276 | 0.8566 | 0.8889 |
| 2 | 16 | 32.9422 | 291.9822 | 0.9026 | 0.9412 |
| 2 | 32 | 32.9422 | 568.2915 | 0.9275 | 0.9697 |
| 4 | 1 | 32.1948 | 32.1948 | 0.2500 | 0.2500 |
| 4 | 2 | 32.1948 | 40.8295 | 0.3943 | 0.4000 |
| 4 | 4 | 32.1948 | 58.0988 | 0.5541 | 0.5714 |
| 4 | 8 | 32.1948 | 92.6375 | 0.6951 | 0.7273 |
| 4 | 16 | 32.1948 | 161.7148 | 0.7963 | 0.8421 |
| 4 | 32 | 32.1948 | 299.8695 | 0.8589 | 0.9143 |
| 8 | 1 | 32.2061 | 32.2061 | 0.1250 | 0.1250 |
| 8 | 2 | 32.2061 | 36.5234 | 0.2204 | 0.2222 |
| 8 | 4 | 32.2061 | 45.1581 | 0.3566 | 0.3636 |
| 8 | 8 | 32.2061 | 62.4274 | 0.5159 | 0.5333 |
| 8 | 16 | 32.2061 | 96.9661 | 0.6643 | 0.6957 |
| 8 | 32 | 32.2061 | 166.0434 | 0.7758 | 0.8205 |

The scheduler is a FIFO flow-shop simulation with one worker per stage and dimensionless work units. It is not a vLLM serving trace, a 1F1B training schedule or a wall-clock benchmark. Stage cost includes deterministic synthetic ±8% speed variation.

## Eight-GPU hybrid layouts

| TP × PP | Layers per stage | FP16 weight GiB/GPU | Max-stage TP ring send MiB/rank | PP activation MiB/rank (assumption) |
|---:|---|---:|---:|---:|
| 8 × 1 | 32 | 1.5078 | 14.0000 | 0.0000 |
| 4 × 2 | 16;16 | 1.5078 | 6.0000 | 0.0312 |
| 2 × 4 | 8;8;8;8 | 1.5078 | 2.0000 | 0.1875 |
| 1 × 8 | 4;4;4;4;4;4;4;4 | 1.5078 | 0.0000 | 0.8750 |

The hybrid table assumes two nodes with four GPUs per node and groups each TP communicator locally. Its stage-boundary activation assumes hidden-state sharding across TP ranks. Real model layouts can keep tensors replicated or change layouts through collectives, so use actual runtime traces for a deployment estimate.

## Collective semantics

- AllReduce: [111, 222, 333, 444] replicated to every rank.
- AllGather: [1, 2, 3, 4, 5, 6] assembled in rank order.
- ReduceScatter: [[111, 222], [333, 444], [555, 666]] after sum and equal partition.
- AllToAll: [[0, 10, 20], [1, 11, 21], [2, 12, 22]] after source/destination exchange.

## Checks

- [x] **PASS** — TP x PP rank product covers the fixed GPU count: All four layouts use exactly eight ranks.
- [x] **PASS** — TP=1 has no inter-rank AllReduce payload: Single-rank collectives require no network transfer.
- [x] **PASS** — Ring AllReduce send volume matches the 2(p-1)/p formula: 1,000 elements, two bytes, four ranks: 3,000 bytes sent per rank.
- [x] **PASS** — TP sharding preserves column-parallel linear output: Concatenating output-column shards matches the unsharded matrix product.
- [x] **PASS** — TP sharding preserves row-parallel linear output: Summing input-row partial products matches the unsharded matrix product.
- [x] **PASS** — AllReduce returns the same summed vector to every rank: Three rank-local vectors are summed element-wise and replicated.
- [x] **PASS** — AllGather concatenates shards in rank order: Each rank receives the ordered concatenation.
- [x] **PASS** — ReduceScatter reduces then distributes equal result chunks: Reduction is followed by an even scatter.
- [x] **PASS** — AllToAll transposes source-to-destination exchanges: Every destination receives one payload from each source.
- [x] **PASS** — Pipeline schedules complete every microbatch: Every FIFO flow-shop schedule reaches the final stage.
- [x] **PASS** — More microbatches improve balanced fill-drain reference occupancy: The ideal reference approaches full occupancy as job count grows.
- [x] **PASS** — Pipeline stage partition handles uneven layer counts: Layer counts are distributed without dropping or duplicating layers.
- [x] **PASS** — Fast illustrative fabric has lower estimated collective time: Same ring model evaluated with explicit hypothetical alpha/BW parameters.
- [x] **PASS** — TP reduces ideal weight payload per rank: Toy-model weight payload is divided evenly over TP ranks.
- [x] **PASS** — Hybrid TP4 x PP2 maps TP groups inside each node: Each four-rank TP group is kept within one node.
- [x] **PASS** — Single-microbatch first-result latency spans every pipeline stage: One microbatch traverses every stage in sequence.

## Interpretation and limits

- The simulator does not import torch, NCCL, CUDA, Ray or vLLM.
- No GPU, PCIe, NVLink, InfiniBand, RoCE or Ethernet link is exercised.
- Fabric A/B are hypothetical alpha/bandwidth sensitivity profiles, not hardware specifications.
- Ring formulas are teaching models; NCCL can choose Ring, Tree or other algorithms based on topology and version.
- Pipeline work units are dimensionless and cannot be interpreted as latency or throughput.
- Model weights exclude embeddings, LM head, norms, quantization metadata, KV Cache and runtime memory.
- Actual communication depends on sharding layouts, fused kernels, sequence parallelism, dimensions, batch and backend.
- For real systems, run NCCL tests, inspect nvidia-smi topology, check P2P and measure the actual workload.

## Evidence files

- evidence/tensor-parallel-communication.csv
- evidence/pipeline-schedule.csv
- evidence/pipeline-stage-utilization.csv
- evidence/pipeline-events-sample.csv
- evidence/hybrid-tp-pp-topology.csv
- evidence/tp4-pp2-rank-map.csv
- evidence/pipeline-activation-transfer.csv
- reports/metrics.json
