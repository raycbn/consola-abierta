# LLM Engineering 07 — Distributed inference concepts

A reproducible, standard-library-only simulator for Tensor Parallelism (TP), Pipeline Parallelism (PP), collective communication and hybrid TP×PP placement.

## Requirements and cost

- Python 3.10+
- Python standard library only
- No PyTorch, CUDA, NCCL, Ray, GPU or model downloads
- Cost: €0

## Run

Windows PowerShell:

    .\run.ps1

Or directly:

    python .\src\run_lab.py

The fixed seed is 20261114. CSV evidence and JSON/Markdown reports are regenerated on each run.

## What is simulated

1. A 32-layer decoder-only-style model with hidden size 4096 and FFN size 11008.
2. Ideal weight payload per rank for TP=1,2,4,8.
3. Ring AllReduce send volume per rank using 2(p−1)/p × N × element_bytes.
4. A simplified latency sensitivity model with two explicitly hypothetical alpha/bandwidth profiles.
5. FIFO pipeline fill/drain schedules for PP=1,2,4,8 and 1–32 independent microbatches.
6. Uneven layer partitioning and per-stage utilization.
7. Eight-GPU hybrid layouts: TP8×PP1, TP4×PP2, TP2×PP4 and TP1×PP8.
8. A rank map for TP4×PP2 across two four-GPU nodes.
9. Pipeline activation-transfer arithmetic.
10. Numeric proof that column-wise and row-wise matrix sharding preserve a linear projection.
11. Semantic examples for AllReduce, AllGather, ReduceScatter and AllToAll.

## Evidence

- evidence/tensor-parallel-communication.csv
- evidence/pipeline-schedule.csv
- evidence/pipeline-stage-utilization.csv
- evidence/pipeline-events-sample.csv
- evidence/hybrid-tp-pp-topology.csv
- evidence/tp4-pp2-rank-map.csv
- evidence/pipeline-activation-transfer.csv
- reports/metrics.json
- reports/lab-report.md

## Reproduced results

- Python 3.12.10
- 16/16 checks PASS
- Toy model weights: 6,476,005,376 parameters (embeddings, LM head, norms, biases and model-specific tensors excluded)
- Ideal FP16/BF16 weight payload: 12.062 GiB
- TP=1/2/4/8 ideal weight payload per rank: 12.062 / 6.031 / 3.016 / 1.508 GiB
- TP=1/2/4/8 ring send volume per rank per decode iteration: 0 / 8 / 12 / 14 MiB
- PP=1/2/4/8, 16 microbatches: aggregate simulated stage utilization 1.000 / 0.903 / 0.796 / 0.664

The work units are dimensionless. The two alpha/bandwidth profiles are hypothetical sensitivity parameters.

## Boundaries — read before interpreting

This is a teaching simulator, not a distributed inference engine.

- It does not launch multiple processes or ranks.
- It does not use PyTorch, NCCL, CUDA, Ray, NVLink, PCIe, InfiniBand or RoCE.
- Ring AllReduce arithmetic is a reference model; NCCL may select Ring, Tree or other algorithms.
- Collective formulas omit topology routing, protocol overhead, synchronization, contention, fusion and compute overlap.
- Pipeline schedules are FIFO flow-shop models, not vLLM traces or a full Megatron schedule.
- Pipeline work units are not milliseconds, TTFT, TPOT or throughput.
- Ideal weight bytes exclude KV Cache, runtime buffers, workspaces, replication, padding and metadata.
- Hybrid activation sharding is an explicit assumption; real layouts may replicate or change layouts at stage boundaries.

For real hardware, inspect topology and P2P, run NCCL Tests / nvbandwidth, check NCCL logs, then benchmark the actual serving workload with realistic prompt lengths, decode batch, concurrency and model configuration.
