#!/usr/bin/env python3
"""LLM Engineering 07 — distributed inference concepts lab.

Pure Python standard library. This is a deterministic communication and
pipeline simulator, NOT NCCL, CUDA, a GPU benchmark, or a distributed runtime.
"""
from __future__ import annotations

import csv
import json
import math
import platform
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
REPORTS = ROOT / "reports"
SEED = 20261114

# Explicit toy workload: a decoder-only Transformer-like stack.
HIDDEN = 4096
FFN = 11008
LAYERS = 32
ACTIVE_SEQUENCES = 16
DTYPE_BYTES = 2  # BF16/FP16 payload assumption.
MICROBATCH_COUNTS = [1, 2, 4, 8, 16, 32]


def ensure_dirs():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError(f"No rows generated for {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def split_even(total: int, parts: int) -> list[int]:
    if parts < 1:
        raise ValueError("parts must be >= 1")
    q, r = divmod(total, parts)
    return [q + (1 if i < r else 0) for i in range(parts)]


def model_parameter_count(hidden: int, ffn: int, layers: int) -> int:
    """Weights only: QKV + attention output + gated up/gate + down per layer.

    Embeddings, norms, biases, output head and architecture-specific tensors
    are excluded to keep the arithmetic auditable.
    """
    per_layer = 3 * hidden * hidden + hidden * hidden + 2 * hidden * ffn + ffn * hidden
    return per_layer * layers


def ring_allreduce_bytes_sent_per_rank(elements: int, element_bytes: int, ranks: int) -> int:
    """Ideal ring payload per rank: 2*(p-1)/p * N elements.

    This is a teaching model. NCCL can select other algorithms/protocols.
    """
    if ranks < 1 or elements < 0 or element_bytes < 1:
        raise ValueError("Invalid collective dimensions")
    if ranks == 1:
        return 0
    return math.ceil(2 * (ranks - 1) / ranks * elements * element_bytes)


def ring_allreduce_time_seconds(elements: int, element_bytes: int, ranks: int,
                                alpha_seconds: float, bandwidth_bytes_per_second: float) -> float:
    """Simplified ring model: 2(p-1)*alpha + bytes_sent/BW."""
    if ranks == 1:
        return 0.0
    if bandwidth_bytes_per_second <= 0:
        raise ValueError("Bandwidth must be positive")
    sent = ring_allreduce_bytes_sent_per_rank(elements, element_bytes, ranks)
    return 2 * (ranks - 1) * alpha_seconds + sent / bandwidth_bytes_per_second


def collective_all_reduce(vectors: list[list[float]]) -> list[list[float]]:
    if not vectors or any(len(v) != len(vectors[0]) for v in vectors):
        raise ValueError("AllReduce requires equal, non-empty vectors")
    reduced = [sum(v[i] for v in vectors) for i in range(len(vectors[0]))]
    return [reduced[:] for _ in vectors]


def collective_all_gather(chunks: list[list[float]]) -> list[list[float]]:
    if not chunks or any(len(v) != len(chunks[0]) for v in chunks):
        raise ValueError("AllGather requires equal, non-empty chunks")
    gathered = [x for chunk in chunks for x in chunk]
    return [gathered[:] for _ in chunks]


def collective_reduce_scatter(vectors: list[list[float]]) -> list[list[float]]:
    if not vectors or any(len(v) != len(vectors[0]) for v in vectors):
        raise ValueError("ReduceScatter requires equal, non-empty vectors")
    ranks, n = len(vectors), len(vectors[0])
    if n % ranks:
        raise ValueError("ReduceScatter input length must divide evenly by ranks")
    reduced = [sum(v[i] for v in vectors) for i in range(n)]
    width = n // ranks
    return [reduced[i * width:(i + 1) * width] for i in range(ranks)]


def collective_all_to_all(matrix: list[list[float]]) -> list[list[float]]:
    """Each source sends one scalar to each destination."""
    if not matrix or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("AllToAll requires a square source/destination matrix")
    ranks = len(matrix)
    return [[matrix[source][destination] for source in range(ranks)]
            for destination in range(ranks)]


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    if not a or not b or len(a[0]) != len(b):
        raise ValueError("matmul shape mismatch")
    if any(len(row) != len(a[0]) for row in a) or any(len(row) != len(b[0]) for row in b):
        raise ValueError("matmul requires rectangular matrices")
    return [[sum(a[i][k] * b[k][j] for k in range(len(b)))
             for j in range(len(b[0]))] for i in range(len(a))]


def concat_columns(parts: list[list[list[float]]]) -> list[list[float]]:
    if not parts:
        raise ValueError("No column shards")
    rows = len(parts[0])
    if any(len(part) != rows for part in parts):
        raise ValueError("Column shards have inconsistent row counts")
    return [[value for part in parts for value in part[i]] for i in range(rows)]


def add_matrices(parts: list[list[list[float]]]) -> list[list[float]]:
    if not parts:
        raise ValueError("No partial matrices")
    rows, cols = len(parts[0]), len(parts[0][0])
    if any(len(part) != rows or any(len(row) != cols for row in part) for part in parts):
        raise ValueError("Partial matrix dimensions differ")
    return [[sum(part[i][j] for part in parts) for j in range(cols)]
            for i in range(rows)]


def max_matrix_error(reference, candidate):
    return max(abs(reference[i][j] - candidate[i][j])
               for i in range(len(reference)) for j in range(len(reference[0])))


def matrix_close(a, b, tolerance=1e-10):
    return len(a) == len(b) and all(
        len(x) == len(y) and all(abs(v - w) <= tolerance for v, w in zip(x, y))
        for x, y in zip(a, b)
    )


def schedule_pipeline(stage_times: list[float], jobs: int) -> dict:
    """FIFO flow-shop schedule, one worker per stage.

    Jobs represent independent request microbatches. This is not a vLLM
    scheduler or a full Megatron training schedule.
    """
    if not stage_times or jobs < 1 or any(t <= 0 for t in stage_times):
        raise ValueError("Pipeline needs positive stage times and at least one job")
    stages = len(stage_times)
    completions = [[0.0] * stages for _ in range(jobs)]
    events = []
    busy = [0.0] * stages
    for job in range(jobs):
        for stage in range(stages):
            ready_from_previous_job = completions[job - 1][stage] if job > 0 else 0.0
            ready_from_previous_stage = completions[job][stage - 1] if stage > 0 else 0.0
            start = max(ready_from_previous_job, ready_from_previous_stage)
            finish = start + stage_times[stage]
            completions[job][stage] = finish
            busy[stage] += stage_times[stage]
            events.append({
                "microbatch": job, "stage": stage,
                "start_work_units": round(start, 6),
                "finish_work_units": round(finish, 6),
                "service_work_units": round(stage_times[stage], 6),
            })
    makespan = completions[-1][-1]
    stage_rows = [{
        "stage": i,
        "service_work_units": stage_times[i],
        "busy_work_units": busy[i],
        "makespan_work_units": makespan,
        "utilization_fraction": busy[i] / makespan if makespan else 0.0,
    } for i in range(stages)]
    return {
        "jobs": jobs, "stages": stages, "makespan": makespan,
        "first_result_latency_work_units": sum(stage_times),
        "completed_jobs": len(completions),
        "aggregate_stage_utilization": sum(busy) / (stages * makespan),
        "stage_rows": stage_rows, "events": events,
    }


def synthetic_matrices(seed: int = SEED):
    rng = random.Random(seed)
    a = [[rng.uniform(-0.8, 0.8) for _ in range(8)] for _ in range(4)]
    b = [[rng.uniform(-0.8, 0.8) for _ in range(8)] for _ in range(8)]
    return a, b


def main():
    ensure_dirs()
    params = model_parameter_count(HIDDEN, FFN, LAYERS)
    weight_bytes_fp16 = params * DTYPE_BYTES
    decode_elements = ACTIVE_SEQUENCES * HIDDEN
    collectives_per_layer = 2  # Canonical row-parallel reductions in this toy block.
    fabric_profiles = [
        {"name": "Illustrative fabric A", "alpha_us": 4.0, "bandwidth_GBps": 200.0},
        {"name": "Illustrative fabric B", "alpha_us": 10.0, "bandwidth_GBps": 50.0},
    ]

    # 1) Tensor-parallel traffic and first-order weight memory.
    tp_rows = []
    for tp in [1, 2, 4, 8]:
        sent_per_collective = ring_allreduce_bytes_sent_per_rank(decode_elements, DTYPE_BYTES, tp)
        ops = LAYERS * collectives_per_layer
        sent_step = sent_per_collective * ops
        row = {
            "tp_degree": tp,
            "active_sequences_decode_tokens": ACTIVE_SEQUENCES,
            "hidden_size": HIDDEN,
            "element_bytes": DTYPE_BYTES,
            "collectives_per_layer_assumption": collectives_per_layer,
            "layers": LAYERS,
            "allreduce_ops_per_decode_iteration": ops if tp > 1 else 0,
            "ring_send_bytes_per_rank_per_allreduce": sent_per_collective,
            "ring_send_MiB_per_rank_per_decode_iteration": round(sent_step / (1024 ** 2), 6),
            "ideal_weight_GiB_per_rank_fp16": round(weight_bytes_fp16 / tp / (1024 ** 3), 6),
            "traffic_model": "ring payload formula; excludes headers, padding, fusion and overlap",
        }
        for profile in fabric_profiles:
            seconds = ring_allreduce_time_seconds(
                decode_elements, DTYPE_BYTES, tp,
                profile["alpha_us"] * 1e-6,
                profile["bandwidth_GBps"] * 1e9,
            )
            row[profile["name"].lower().replace(" ", "_") + "_ms_per_decode_iteration"] = round(
                seconds * ops * 1000, 6
            )
        tp_rows.append(row)

    # 2) Pipeline scheduling and stage balance.
    pp_rows, stage_rows, schedule_events = [], [], []
    for pp in [1, 2, 4, 8]:
        layer_counts = split_even(LAYERS, pp)
        stage_factors = [1.0 + 0.08 * math.sin((stage + 1) * 1.7) for stage in range(pp)]
        stage_times = [count * factor for count, factor in zip(layer_counts, stage_factors)]
        for jobs in MICROBATCH_COUNTS:
            sim = schedule_pipeline(stage_times, jobs)
            pp_rows.append({
                "pp_degree": pp, "microbatches": jobs,
                "layers_per_stage": ";".join(map(str, layer_counts)),
                "stage_service_work_units": ";".join(f"{x:.6f}" for x in stage_times),
                "first_result_latency_work_units": round(sim["first_result_latency_work_units"], 6),
                "makespan_work_units": round(sim["makespan"], 6),
                "completed_microbatches": sim["completed_jobs"],
                "aggregate_stage_utilization": round(sim["aggregate_stage_utilization"], 8),
                "ideal_balanced_fill_drain_utilization": round(jobs / (jobs + pp - 1), 8),
                "note": "synthetic FIFO pipeline; work units are not milliseconds",
            })
            for stage_row in sim["stage_rows"]:
                stage_rows.append({"pp_degree": pp, "microbatches": jobs, **stage_row})
            if pp == 4 and jobs in {1, 4, 16}:
                for event in sim["events"]:
                    schedule_events.append({"pp_degree": pp, "microbatches": jobs, **event})

    # 3) Hybrid TP x PP layouts across eight GPUs and two nodes.
    hybrid_rows, rank_rows = [], []
    fixed_gpus, nodes, gpus_per_node = 8, 2, 4
    for tp, pp in [(8, 1), (4, 2), (2, 4), (1, 8)]:
        if tp * pp != fixed_gpus:
            raise AssertionError("TP x PP must account for all GPUs")
        layers_by_stage = split_even(LAYERS, pp)
        per_rank_tp_sent = ring_allreduce_bytes_sent_per_rank(decode_elements, DTYPE_BYTES, tp)
        max_stage_collectives = max(layers_by_stage) * collectives_per_layer if tp > 1 else 0
        per_rank_sent_max_stage = per_rank_tp_sent * max_stage_collectives
        activation_bytes_total = ACTIVE_SEQUENCES * HIDDEN * DTYPE_BYTES
        activation_bytes_per_rank_boundary = math.ceil(activation_bytes_total / tp)
        hybrid_rows.append({
            "total_gpus": fixed_gpus, "nodes": nodes, "gpus_per_node": gpus_per_node,
            "tp_degree": tp, "pp_degree": pp,
            "layers_by_pipeline_stage": ";".join(map(str, layers_by_stage)),
            "ideal_weight_GiB_per_gpu_fp16": round(weight_bytes_fp16 / fixed_gpus / (1024 ** 3), 6),
            "max_stage_tp_collectives_per_rank_per_decode_iteration": max_stage_collectives,
            "ring_send_MiB_per_rank_at_max_pipeline_stage": round(per_rank_sent_max_stage / (1024 ** 2), 6),
            "pp_activation_bytes_per_rank_per_boundary_assumption": activation_bytes_per_rank_boundary if pp > 1 else 0,
            "pp_activation_MiB_per_rank_all_boundaries": round(
                activation_bytes_per_rank_boundary * (pp - 1) / (1024 ** 2), 6
            ),
            "layout_note": "illustrative rank grouping; actual activation layout may be replicated or sharded differently",
        })
        if tp == 4 and pp == 2:
            for stage in range(pp):
                for tp_rank in range(tp):
                    global_rank = stage * tp + tp_rank
                    rank_rows.append({
                        "global_rank": global_rank, "pipeline_stage": stage,
                        "tensor_rank": tp_rank, "node": stage,
                        "local_gpu": tp_rank, "group": f"pp{stage}-tp",
                    })

    # 4) PP-only activation transfer arithmetic.
    pp_transfer_rows = []
    for pp in [1, 2, 4, 8]:
        boundary_bytes = ACTIVE_SEQUENCES * HIDDEN * DTYPE_BYTES
        pp_transfer_rows.append({
            "pp_degree": pp, "pipeline_boundaries": pp - 1,
            "active_sequences": ACTIVE_SEQUENCES, "hidden_size": HIDDEN,
            "element_bytes": DTYPE_BYTES,
            "payload_bytes_per_boundary_per_decode_iteration": boundary_bytes if pp > 1 else 0,
            "aggregate_payload_MiB_across_boundaries": round(boundary_bytes * (pp - 1) / (1024 ** 2), 6),
            "assumption": "full hidden activation crosses each boundary; no TP sharding or transport overhead",
        })

    # 5) Numeric proof of column-wise and row-wise linear sharding.
    a, b = synthetic_matrices()
    baseline = matmul(a, b)
    ranks = 4
    col_width = len(b[0]) // ranks
    col_shards = []
    for rank in range(ranks):
        shard = [row[rank * col_width:(rank + 1) * col_width] for row in b]
        col_shards.append(matmul(a, shard))
    col_result = concat_columns(col_shards)

    row_width = len(b) // ranks
    row_partial = []
    for rank in range(ranks):
        a_shard = [row[rank * row_width:(rank + 1) * row_width] for row in a]
        b_shard = b[rank * row_width:(rank + 1) * row_width]
        row_partial.append(matmul(a_shard, b_shard))
    row_result = add_matrices(row_partial)

    # 6) Small deterministic collective-semantic examples.
    reduce_input = [[1, 2, 3, 4], [10, 20, 30, 40], [100, 200, 300, 400]]
    allreduce_result = collective_all_reduce(reduce_input)
    gather_result = collective_all_gather([[1, 2], [3, 4], [5, 6]])
    scatter_result = collective_reduce_scatter([
        [1, 2, 3, 4, 5, 6], [10, 20, 30, 40, 50, 60], [100, 200, 300, 400, 500, 600],
    ])
    alltoall_result = collective_all_to_all([[0, 1, 2], [10, 11, 12], [20, 21, 22]])

    checks = []

    def check(name, passed, detail):
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "detail": detail})

    check("TP x PP rank product covers the fixed GPU count",
          all(row["tp_degree"] * row["pp_degree"] == fixed_gpus for row in hybrid_rows),
          "All four layouts use exactly eight ranks.")
    check("TP=1 has no inter-rank AllReduce payload",
          tp_rows[0]["ring_send_bytes_per_rank_per_allreduce"] == 0,
          "Single-rank collectives require no network transfer.")
    check("Ring AllReduce send volume matches the 2(p-1)/p formula",
          ring_allreduce_bytes_sent_per_rank(1000, 2, 4) == 3000,
          "1,000 elements, two bytes, four ranks: 3,000 bytes sent per rank.")
    check("TP sharding preserves column-parallel linear output",
          matrix_close(baseline, col_result),
          "Concatenating output-column shards matches the unsharded matrix product.")
    check("TP sharding preserves row-parallel linear output",
          matrix_close(baseline, row_result),
          "Summing input-row partial products matches the unsharded matrix product.")
    check("AllReduce returns the same summed vector to every rank",
          all(v == [111, 222, 333, 444] for v in allreduce_result),
          "Three rank-local vectors are summed element-wise and replicated.")
    check("AllGather concatenates shards in rank order",
          all(v == [1, 2, 3, 4, 5, 6] for v in gather_result),
          "Each rank receives the ordered concatenation.")
    check("ReduceScatter reduces then distributes equal result chunks",
          scatter_result == [[111, 222], [333, 444], [555, 666]],
          "Reduction is followed by an even scatter.")
    check("AllToAll transposes source-to-destination exchanges",
          alltoall_result == [[0, 10, 20], [1, 11, 21], [2, 12, 22]],
          "Every destination receives one payload from each source.")
    check("Pipeline schedules complete every microbatch",
          all(row["completed_microbatches"] == row["microbatches"] for row in pp_rows),
          "Every FIFO flow-shop schedule reaches the final stage.")
    pp4_rows = [r for r in pp_rows if r["pp_degree"] == 4]
    check("More microbatches improve balanced fill-drain reference occupancy",
          all(b["ideal_balanced_fill_drain_utilization"] >= a["ideal_balanced_fill_drain_utilization"]
              for a, b in zip(pp4_rows[:-1], pp4_rows[1:])),
          "The ideal reference approaches full occupancy as job count grows.")
    check("Pipeline stage partition handles uneven layer counts",
          split_even(32, 3) == [11, 11, 10] and split_even(32, 8) == [4] * 8,
          "Layer counts are distributed without dropping or duplicating layers.")
    check("Fast illustrative fabric has lower estimated collective time",
          all(row["illustrative_fabric_a_ms_per_decode_iteration"] <
              row["illustrative_fabric_b_ms_per_decode_iteration"] for row in tp_rows if row["tp_degree"] > 1),
          "Same ring model evaluated with explicit hypothetical alpha/BW parameters.")
    check("TP reduces ideal weight payload per rank",
          tp_rows[-1]["ideal_weight_GiB_per_rank_fp16"] < tp_rows[0]["ideal_weight_GiB_per_rank_fp16"],
          "Toy-model weight payload is divided evenly over TP ranks.")
    check("Hybrid TP4 x PP2 maps TP groups inside each node",
          len(rank_rows) == 8 and all(x["node"] == x["pipeline_stage"] and x["local_gpu"] == x["tensor_rank"] for x in rank_rows),
          "Each four-rank TP group is kept within one node.")
    check("Single-microbatch first-result latency spans every pipeline stage",
          all(math.isclose(r["first_result_latency_work_units"],
                           sum(float(x) for x in r["stage_service_work_units"].split(";")), rel_tol=1e-6)
              for r in pp_rows if r["microbatches"] == 1),
          "One microbatch traverses every stage in sequence.")

    write_csv(EVIDENCE / "tensor-parallel-communication.csv", tp_rows)
    write_csv(EVIDENCE / "pipeline-schedule.csv", pp_rows)
    write_csv(EVIDENCE / "pipeline-stage-utilization.csv", stage_rows)
    write_csv(EVIDENCE / "pipeline-events-sample.csv", schedule_events)
    write_csv(EVIDENCE / "hybrid-tp-pp-topology.csv", hybrid_rows)
    write_csv(EVIDENCE / "tp4-pp2-rank-map.csv", rank_rows)
    write_csv(EVIDENCE / "pipeline-activation-transfer.csv", pp_transfer_rows)

    report = {
        "lab": "LLM Engineering 07 — Distributed inference concepts",
        "date": "2026-11-14",
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "dependencies": "Python standard library only", "seed": SEED,
                        "execution": "single-process CPU simulation"},
        "model_assumptions": {
            "hidden_size": HIDDEN, "ffn_intermediate_size": FFN, "transformer_layers": LAYERS,
            "active_sequences": ACTIVE_SEQUENCES, "element_bytes": DTYPE_BYTES,
            "parameter_count_formula": "layers * (3*h*h + h*h + 2*h*ffn + ffn*h)",
            "parameter_count_estimate": params,
            "excluded_parameters": ["token embeddings", "final norm", "LM head", "biases", "model-specific tensors"],
        },
        "communication_assumptions": {
            "tp_collectives_per_layer": collectives_per_layer,
            "ring_send_formula": "2 * (p - 1) / p * N * element_bytes",
            "ring_time_formula": "2 * (p - 1) * alpha + bytes_sent_per_rank / bandwidth",
            "fabric_profiles": fabric_profiles,
            "profiles_are": "hypothetical sensitivity inputs; not vendor link specifications or measurements",
            "pipeline_activation": "one BF16-sized hidden activation per active sequence per boundary",
        },
        "pipeline_assumptions": {
            "schedule": "FIFO flow-shop; one worker per stage; independent microbatches",
            "stage_costs": "layers per stage plus deterministic ±8% synthetic stage-speed variation",
            "work_units": "dimensionless, not milliseconds",
            "microbatch_counts": MICROBATCH_COUNTS,
        },
        "summary": {"checks_passed": sum(x["status"] == "PASS" for x in checks),
                    "checks_total": len(checks), "tensor_parallel_rows": len(tp_rows),
                    "pipeline_schedule_rows": len(pp_rows), "hybrid_topology_rows": len(hybrid_rows),
                    "pipeline_event_rows": len(schedule_events)},
        "tensor_parallel": tp_rows, "pipeline_schedules": pp_rows,
        "pipeline_stage_utilization": stage_rows, "hybrid_tp_pp_topology": hybrid_rows,
        "tp4_pp2_rank_map": rank_rows, "pipeline_activation_transfer": pp_transfer_rows,
        "collective_examples": {"all_reduce": allreduce_result, "all_gather": gather_result,
                                "reduce_scatter": scatter_result, "all_to_all": alltoall_result},
        "matrix_sharding_validation": {
            "column_parallel_max_abs_error": max_matrix_error(baseline, col_result),
            "row_parallel_max_abs_error": max_matrix_error(baseline, row_result),
            "column_parallel_matches": matrix_close(baseline, col_result),
            "row_parallel_matches": matrix_close(baseline, row_result),
        },
        "checks": checks,
        "interpretation": [
            "Tensor-parallel communication grows with rank count in the simplified ring model, while per-rank weight payload falls.",
            "Pipeline parallelism reduces layers resident per stage but adds fill/drain and activation transfers.",
            "More microbatches improve pipeline occupancy in this simple flow-shop model; real serving schedules are more complex.",
            "TP x PP is a first-order placement model; actual tensors may be replicated, sharded or gathered at stage boundaries.",
            "No real GPU, NCCL, NVLink, PCIe, RDMA, Ray or vLLM operation is executed.",
            "Fabric parameters are hypothetical sensitivity inputs, not hardware performance claims.",
        ],
    }
    (REPORTS / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Lab report — Distributed LLM inference concepts", "",
        "## Reproducibility", "- Date: 2026-11-14", f"- Python: {platform.python_version()}",
        "- Dependencies: Python standard library only", f"- Seed: {SEED}",
        f"- Checks: {report['summary']['checks_passed']}/{report['summary']['checks_total']} PASS",
        "- Execution: single-process CPU simulation; no GPU or distributed processes", "",
        "## Model arithmetic", f"- Hidden size: {HIDDEN}", f"- FFN intermediate size: {FFN}",
        f"- Transformer layers: {LAYERS}", f"- Active sequences: {ACTIVE_SEQUENCES}",
        f"- Toy layer parameter count (weights only): {params:,}",
        f"- Ideal FP16/BF16 weight payload: {weight_bytes_fp16 / (1024**3):.4f} GiB",
        "- Excluded: embeddings, LM head, norms, biases, model-specific tensors, padding, runtime buffers and KV Cache", "",
        "## Tensor Parallelism — communication and memory", "",
        "| TP ranks | Ideal weight GiB/rank | Ring send MiB/rank/decode iteration | Fabric A ms (illustrative) | Fabric B ms (illustrative) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in tp_rows:
        lines.append(f"| {row['tp_degree']} | {row['ideal_weight_GiB_per_rank_fp16']:.4f} | "
                     f"{row['ring_send_MiB_per_rank_per_decode_iteration']:.4f} | "
                     f"{row['illustrative_fabric_a_ms_per_decode_iteration']:.6f} | "
                     f"{row['illustrative_fabric_b_ms_per_decode_iteration']:.6f} |")
    lines += [
        "",
        "The communication model assumes two row-parallel reductions per Transformer layer and a ring AllReduce. It computes 2(p−1)/p × N × bytes sent per rank per collective. Fabric A/B use hypothetical alpha and bandwidth values to demonstrate sensitivity. Times are not GPU/NCCL benchmarks and omit overlap, topology routing, protocol, kernel scheduling and contention.",
        "", "## Pipeline Parallelism — fill/drain schedule", "",
        "| PP stages | Microbatches | First result (work units) | Makespan (work units) | Aggregate stage utilization | Ideal balanced fill/drain reference |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pp_rows:
        lines.append(f"| {row['pp_degree']} | {row['microbatches']} | "
                     f"{row['first_result_latency_work_units']:.4f} | {row['makespan_work_units']:.4f} | "
                     f"{row['aggregate_stage_utilization']:.4f} | {row['ideal_balanced_fill_drain_utilization']:.4f} |")
    lines += [
        "",
        "The scheduler is a FIFO flow-shop simulation with one worker per stage and dimensionless work units. It is not a vLLM serving trace, a 1F1B training schedule or a wall-clock benchmark. Stage cost includes deterministic synthetic ±8% speed variation.",
        "", "## Eight-GPU hybrid layouts", "",
        "| TP × PP | Layers per stage | FP16 weight GiB/GPU | Max-stage TP ring send MiB/rank | PP activation MiB/rank (assumption) |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in hybrid_rows:
        lines.append(f"| {row['tp_degree']} × {row['pp_degree']} | {row['layers_by_pipeline_stage']} | "
                     f"{row['ideal_weight_GiB_per_gpu_fp16']:.4f} | "
                     f"{row['ring_send_MiB_per_rank_at_max_pipeline_stage']:.4f} | "
                     f"{row['pp_activation_MiB_per_rank_all_boundaries']:.4f} |")
    lines += [
        "",
        "The hybrid table assumes two nodes with four GPUs per node and groups each TP communicator locally. Its stage-boundary activation assumes hidden-state sharding across TP ranks. Real model layouts can keep tensors replicated or change layouts through collectives, so use actual runtime traces for a deployment estimate.",
        "", "## Collective semantics", "",
        f"- AllReduce: {allreduce_result[0]} replicated to every rank.",
        f"- AllGather: {gather_result[0]} assembled in rank order.",
        f"- ReduceScatter: {scatter_result} after sum and equal partition.",
        f"- AllToAll: {alltoall_result} after source/destination exchange.",
        "", "## Checks", "",
    ]
    for item in checks:
        lines.append(f"- [{'x' if item['status'] == 'PASS' else ' '}] **{item['status']}** — {item['check']}: {item['detail']}")
    lines += [
        "", "## Interpretation and limits", "",
        "- The simulator does not import torch, NCCL, CUDA, Ray or vLLM.",
        "- No GPU, PCIe, NVLink, InfiniBand, RoCE or Ethernet link is exercised.",
        "- Fabric A/B are hypothetical alpha/bandwidth sensitivity profiles, not hardware specifications.",
        "- Ring formulas are teaching models; NCCL can choose Ring, Tree or other algorithms based on topology and version.",
        "- Pipeline work units are dimensionless and cannot be interpreted as latency or throughput.",
        "- Model weights exclude embeddings, LM head, norms, quantization metadata, KV Cache and runtime memory.",
        "- Actual communication depends on sharding layouts, fused kernels, sequence parallelism, dimensions, batch and backend.",
        "- For real systems, run NCCL tests, inspect nvidia-smi topology, check P2P and measure the actual workload.",
        "", "## Evidence files", "",
        "- evidence/tensor-parallel-communication.csv",
        "- evidence/pipeline-schedule.csv",
        "- evidence/pipeline-stage-utilization.csv",
        "- evidence/pipeline-events-sample.csv",
        "- evidence/hybrid-tp-pp-topology.csv",
        "- evidence/tp4-pp2-rank-map.csv",
        "- evidence/pipeline-activation-transfer.csv",
        "- reports/metrics.json",
    ]
    (REPORTS / "lab-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Lab: LLM Engineering 07 — Distributed inference concepts")
    print(f"Python: {platform.python_version()} | standard library only | seed={SEED}")
    print(f"Toy model parameters (weights only): {params:,} | FP16/BF16 payload={weight_bytes_fp16 / (1024**3):.3f} GiB")
    print("\nTensor Parallel — communication sensitivity:")
    for row in tp_rows:
        print(f"  TP={row['tp_degree']:<2} weights/rank={row['ideal_weight_GiB_per_rank_fp16']:.3f} GiB "
              f"send/rank/step={row['ring_send_MiB_per_rank_per_decode_iteration']:.3f} MiB")
    print("\nPipeline Parallel — 16 microbatches:")
    for row in pp_rows:
        if row["microbatches"] == 16:
            print(f"  PP={row['pp_degree']:<2} makespan={row['makespan_work_units']:.3f} work units "
                  f"utilization={row['aggregate_stage_utilization']:.3f}")
    print(f"\nChecks: {report['summary']['checks_passed']}/{report['summary']['checks_total']} PASS")
    for item in checks:
        print(f"  {item['status']}: {item['check']}")
    print(f"\nEvidence: {EVIDENCE}\nReports: {REPORTS}")
    return 0 if all(x["status"] == "PASS" for x in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
