#!/usr/bin/env python3
"""LLM Engineering 06 — quantization fundamentals lab.

Pure Python standard library. Deterministic numerical emulation only:
no model weights, no tokenizer, no GPU, no CUDA and no inference benchmark.
"""
from __future__ import annotations

import csv
import json
import math
import platform
import random
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
REPORTS = ROOT / "reports"
SEED = 20261114


def ensure_dirs():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)


def flatten(matrix):
    return [value for row in matrix for value in row]


def clamp(value, low, high):
    return max(low, min(high, value))


def quantize_regions(rows, bits, granularity, group_size=None, symmetric=True):
    """RTN linear quantization; channel means one output row of [out, in]."""
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("Input must be a non-empty rectangular matrix")
    if bits < 2 or bits > 16:
        raise ValueError("bits must be between 2 and 16")
    if granularity not in {"tensor", "channel", "group"}:
        raise ValueError("granularity must be tensor, channel or group")
    if granularity == "group" and (not group_size or group_size < 1):
        raise ValueError("group_size is required for group granularity")

    row_count, col_count = len(rows), len(rows[0])
    q_rows = [[0 for _ in row] for row in rows]
    dq_rows = [[0.0 for _ in row] for row in rows]
    regions = []
    if granularity == "tensor":
        regions = [[(r, c) for r in range(row_count) for c in range(col_count)]]
    elif granularity == "channel":
        regions = [[(r, c) for c in range(col_count)] for r in range(row_count)]
    else:
        for r in range(row_count):
            for start in range(0, col_count, group_size):
                regions.append([(r, c) for c in range(start, min(start + group_size, col_count))])

    if symmetric:
        qmax = (1 << (bits - 1)) - 1
        qmin = -qmax
    else:
        qmin, qmax = 0, (1 << bits) - 1

    params = []
    for coordinates in regions:
        values = [rows[r][c] for r, c in coordinates]
        if symmetric:
            max_abs = max(abs(value) for value in values)
            scale = max_abs / qmax if max_abs else 1.0
            zero_point = 0
        else:
            observed_min, observed_max = min(values), max(values)
            scale = (observed_max - observed_min) / (qmax - qmin)
            if scale == 0:
                scale = 1.0
            zero_point = clamp(round(qmin - observed_min / scale), qmin, qmax)
        for r, c in coordinates:
            q = clamp(round(rows[r][c] / scale + zero_point), qmin, qmax)
            q_rows[r][c] = q
            dq_rows[r][c] = (q - zero_point) * scale
        params.append({"scale": scale, "zero_point": zero_point, "elements": len(coordinates)})

    return {
        "q": q_rows, "dequant": dq_rows, "parameters": params,
        "bits": bits, "granularity": granularity, "group_size": group_size,
        "symmetric": symmetric, "qmin": qmin, "qmax": qmax,
    }


def quantize_vector(values, bits=8, symmetric=True):
    return quantize_regions([values], bits, "tensor", symmetric=symmetric)["dequant"][0]


def metrics(reference, approximation):
    if len(reference) != len(approximation) or not reference:
        raise ValueError("Metric vectors must have equal non-zero lengths")
    errors = [a - b for a, b in zip(reference, approximation)]
    mse = sum(error * error for error in errors) / len(errors)
    rmse = math.sqrt(mse)
    mae = sum(abs(error) for error in errors) / len(errors)
    max_abs = max(abs(error) for error in errors)
    signal = sum(value * value for value in reference)
    noise = sum(error * error for error in errors)
    sqnr = 10.0 * math.log10(signal / noise) if noise else float("inf")
    relative_l2 = math.sqrt(noise / signal) if signal else 0.0
    return {"mae": mae, "mse": mse, "rmse": rmse,
            "max_abs_error": max_abs, "sqnr_db": sqnr, "relative_l2": relative_l2}


def matrix_metrics(reference, approximation):
    return metrics(flatten(reference), flatten(approximation))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows generated for {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values, pct):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires values")
    position = (len(ordered) - 1) * pct / 100.0
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def fp16_round(value):
    try:
        return struct.unpack("<e", struct.pack("<e", value))[0]
    except OverflowError:
        return math.copysign(float("inf"), value)


def bf16_round(value):
    """Round IEEE float32 to bfloat16 with round-to-nearest, ties-to-even."""
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    exponent = bits & 0x7F800000
    if exponent == 0x7F800000:
        rounded = bits & 0xFFFF0000
    elif (bits & 0x7FFFFFFF) == 0:
        rounded = bits
    else:
        upper = bits >> 16
        bias = 0x7FFF + (upper & 1)
        rounded = (bits + bias) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", rounded))[0]


def make_weights():
    rng = random.Random(SEED)
    result = []
    for r in range(32):
        row_scale = 0.12 + (r % 8) * 0.27
        row = [rng.gauss(0.0, row_scale) + 0.025 * math.sin((r + 1) * (c + 1) / 9.0)
               for c in range(64)]
        result.append(row)
    result[5][19], result[22][41], result[11][7] = 10.5, -9.2, 5.8
    return result


def make_activations():
    rng = random.Random(SEED + 1)
    vectors = []
    for index in range(24):
        vector = [rng.gauss(0.0, 0.65) + 0.08 * math.sin((index + 1) * (c + 1) / 13.0)
                  for c in range(64)]
        if index in {3, 11, 19}:
            vector[(index * 7) % 64] = 13.0 * (-1.0 if index == 11 else 1.0)
        vectors.append(vector)
    return vectors


def matvec(weights, vector):
    if any(len(row) != len(vector) for row in weights):
        raise ValueError("matvec shape mismatch")
    return [sum(weight * value for weight, value in zip(row, vector)) for row in weights]


def quantize_activation(values, strategy):
    absolute = [abs(value) for value in values]
    threshold = max(absolute) if strategy == "absmax" else percentile(absolute, 99.0)
    scale = threshold / 127.0 if threshold else 1.0
    clipped, output = 0, []
    for value in values:
        limited = max(-threshold, min(threshold, value))
        clipped += int(abs(value) > threshold)
        q = clamp(round(limited / scale), -127, 127)
        output.append(q * scale)
    return output, scale, clipped, threshold


def main():
    ensure_dirs()
    weights, activations = make_weights(), make_activations()
    rows, cols = len(weights), len(weights[0])
    total_weights = rows * cols

    schemes = [
        ("FP16 emulation", None, None, None, "fp16"),
        ("BF16 emulation", None, None, None, "bf16"),
        ("INT8 symmetric per-tensor", 8, "tensor", None, "sym"),
        ("INT8 symmetric per-channel", 8, "channel", None, "sym"),
        ("INT4 symmetric per-tensor", 4, "tensor", None, "sym"),
        ("INT4 symmetric per-channel", 4, "channel", None, "sym"),
        ("INT4 symmetric per-group-16", 4, "group", 16, "sym"),
        ("INT4 symmetric per-group-32", 4, "group", 32, "sym"),
        ("INT4 asymmetric per-tensor", 4, "tensor", None, "asym"),
    ]

    quantized, weight_rows = {}, []
    for name, bits, granularity, group_size, mode in schemes:
        if mode == "fp16":
            dequant, scale_count, effective_bits = [[fp16_round(x) for x in row] for row in weights], 0, 16.0
            qmin, qmax = "", ""
        elif mode == "bf16":
            dequant, scale_count, effective_bits = [[bf16_round(x) for x in row] for row in weights], 0, 16.0
            qmin, qmax = "", ""
        else:
            result = quantize_regions(weights, bits, granularity, group_size, symmetric=(mode == "sym"))
            dequant = result["dequant"]
            scale_count = len(result["parameters"])
            effective_bits = bits + 32.0 * scale_count / total_weights
            qmin, qmax = result["qmin"], result["qmax"]
            quantized[name] = result
            assert all(qmin <= q <= qmax for row in result["q"] for q in row)
        score = matrix_metrics(weights, dequant)
        weight_rows.append({
            "scheme": name, "shape_out_in": f"{rows}x{cols}",
            "nominal_bits_per_weight": 16 if bits is None else bits,
            "granularity": granularity or "element", "group_size": group_size or "",
            "scale_count": scale_count,
            "effective_bits_per_weight_estimate": round(effective_bits, 6),
            "scale_storage_assumption": "FP32 per scale; no zero-point metadata" if scale_count else "not applicable",
            "qmin": qmin, "qmax": qmax,
            **{key: round(value, 10) for key, value in score.items()},
        })

    reference_outputs = [matvec(weights, vector) for vector in activations]
    group16 = quantized["INT4 symmetric per-group-16"]["dequant"]
    channel8 = quantized["INT8 symmetric per-channel"]["dequant"]
    outputs_w4a16 = [matvec(group16, vector) for vector in activations]
    outputs_w8a8 = [matvec(channel8, quantize_vector(vector, 8)) for vector in activations]
    output_rows = []
    for label, output in [
        ("FP32 reference", reference_outputs),
        ("W4A16 educational: INT4 group-16 weights, FP32 activations", outputs_w4a16),
        ("W8A8 educational: INT8 per-channel weights, INT8 per-vector activations", outputs_w8a8),
    ]:
        score = metrics(flatten(reference_outputs), flatten(output))
        output_rows.append({
            "path": label, "input_vectors": len(activations), "output_features": rows,
            **{key: round(value, 10) for key, value in score.items()},
        })

    flat_acts = flatten(activations)
    calibration_rows = []
    for strategy in ("absmax", "percentile_99"):
        approximation, scale, clipped, threshold = quantize_activation(flat_acts, strategy)
        score = metrics(flat_acts, approximation)
        calibration_rows.append({
            "strategy": strategy, "quantizer": "symmetric INT8 educational",
            "threshold": round(threshold, 10), "scale": round(scale, 12),
            "clipped_values": clipped, "total_values": len(flat_acts),
            "clipped_fraction": round(clipped / len(flat_acts), 8),
            **{key: round(value, 10) for key, value in score.items()},
        })

    dtype_values = [0.1, 0.3333333, 1.0, -2.71828, 65504.0, 70000.0, 1e-5, 1e5, -1e5, 1e-8]
    dtype_rows = []
    for value in dtype_values:
        f16, bf16 = fp16_round(value), bf16_round(value)
        dtype_rows.append({
            "input_fp32": value, "fp16_emulated": f16, "fp16_finite": math.isfinite(f16),
            "fp16_abs_error": abs(f16 - value) if math.isfinite(f16) else "",
            "bf16_emulated": bf16, "bf16_finite": math.isfinite(bf16),
            "bf16_abs_error": abs(bf16 - value) if math.isfinite(bf16) else "",
        })

    # Exact algebraic rescaling, inspired by the invariance used in smoothing transforms.
    amax_x = [max(abs(v[c]) for v in activations) for c in range(cols)]
    amax_w = [max(abs(weights[r][c]) for r in range(rows)) for c in range(cols)]
    alpha = 0.5
    smooth_scales = [(amax_x[c] ** alpha) / (amax_w[c] ** (1.0 - alpha) + 1e-12) for c in range(cols)]
    transformed_weights = [[weights[r][c] * smooth_scales[c] for c in range(cols)] for r in range(rows)]
    transformed_outputs = [
        matvec(transformed_weights, [vector[c] / smooth_scales[c] for c in range(cols)])
        for vector in activations
    ]
    transform_score = metrics(flatten(reference_outputs), flatten(transformed_outputs))

    # Ideal payload arithmetic only, no runtime/allocator measurement.
    parameter_count = 7_000_000_000
    memory_rows = []
    for label, bits in [
        ("FP32", 32), ("FP16", 16), ("BF16", 16), ("FP8", 8),
        ("INT8", 8), ("INT4 raw payload", 4),
        ("INT4 + FP32 scale per 64 weights", 4 + 32 / 64),
    ]:
        byte_count = parameter_count * bits / 8
        memory_rows.append({
            "model_parameters": parameter_count, "representation": label,
            "nominal_or_effective_bits_per_parameter": bits,
            "decimal_GB": round(byte_count / 1_000_000_000, 6),
            "binary_GiB": round(byte_count / (1024 ** 3), 6),
            "assumption": "ideal payload; excludes padding, unquantized modules, runtime buffers, KV cache and allocator overhead",
        })

    layers, batch, tokens, kv_heads, head_dim = 32, 16, 8192, 8, 128
    kv_rows = []
    for label, bytes_per_element, metadata in [
        ("FP16/BF16 payload", 2, "none included"),
        ("FP8 payload", 1, "scale metadata excluded"),
        ("INT8 payload", 1, "scale and zero-point metadata excluded"),
        ("4-bit payload", 0.5, "packing and scale metadata excluded"),
    ]:
        byte_count = 2 * layers * batch * tokens * kv_heads * head_dim * bytes_per_element
        kv_rows.append({
            "layers": layers, "batch_sequences": batch, "tokens_per_sequence": tokens,
            "kv_heads": kv_heads, "head_dim": head_dim, "representation": label,
            "ideal_payload_GiB": round(byte_count / (1024 ** 3), 6),
            "metadata_assumption": metadata,
            "formula": "2 * layers * batch * tokens * kv_heads * head_dim * bytes_per_element",
        })

    checks = []
    def check(name, passed, detail):
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "detail": detail})

    by_scheme = {row["scheme"]: row for row in weight_rows}
    check("Per-channel INT4 beats per-tensor INT4 on heterogeneous rows",
          by_scheme["INT4 symmetric per-channel"]["mse"] < by_scheme["INT4 symmetric per-tensor"]["mse"],
          f"per-channel={by_scheme['INT4 symmetric per-channel']['mse']}; per-tensor={by_scheme['INT4 symmetric per-tensor']['mse']}")
    check("Groupwise INT4 improves on per-tensor INT4 for this matrix",
          by_scheme["INT4 symmetric per-group-16"]["mse"] < by_scheme["INT4 symmetric per-tensor"]["mse"],
          f"group-16={by_scheme['INT4 symmetric per-group-16']['mse']}; per-tensor={by_scheme['INT4 symmetric per-tensor']['mse']}")
    check("INT8 has lower weight reconstruction MSE than INT4",
          by_scheme["INT8 symmetric per-channel"]["mse"] < by_scheme["INT4 symmetric per-channel"]["mse"],
          "Same synthetic matrix and per-channel granularity.")
    check("BF16 preserves a large finite value that overflows FP16",
          math.isinf(fp16_round(100000.0)) and math.isfinite(bf16_round(100000.0)),
          f"FP16(100000)={fp16_round(100000.0)}; BF16(100000)={bf16_round(100000.0)}")
    affine = quantize_regions([[0.0, 1.0, 2.0]], 4, "tensor", symmetric=False)
    check("Affine quantization represents real zero through zero-point", affine["dequant"][0][0] == 0.0,
          "Asymmetric affine INT4 maps real zero exactly.")
    check("INT4 q values remain inside the configured range",
          all(-7 <= q <= 7 for row in quantized["INT4 symmetric per-group-16"]["q"] for q in row),
          "Symmetric INT4 uses the explicit educational range [-7, 7].")
    check("Algebraic channel rescaling preserves linear output", transform_score["max_abs_error"] < 1e-9,
          f"maximum absolute output drift={transform_score['max_abs_error']}")
    check("W4A16 and W8A8 paths produce finite output errors",
          all(math.isfinite(row["rmse"]) for row in output_rows[1:]),
          "Synthetic numerical paths only; no hardware kernel.")
    check("Memory arithmetic halves nominal payload from FP16 to FP8",
          memory_rows[1]["decimal_GB"] == 14.0 and memory_rows[3]["decimal_GB"] == 7.0,
          "7B parameters: FP16=14 GB decimal; FP8=7 GB decimal, before overhead.")
    check("KV payload arithmetic halves from 16-bit to 8-bit",
          kv_rows[0]["ideal_payload_GiB"] == 16.0 and kv_rows[1]["ideal_payload_GiB"] == 8.0,
          "GQA example; scale metadata excluded.")
    check("Percentile activation calibration reports clipped values",
          calibration_rows[1]["clipped_values"] > 0,
          f"Clipped {calibration_rows[1]['clipped_values']} of {calibration_rows[1]['total_values']}.")
    check("All six evidence tables contain rows",
          all([weight_rows, output_rows, calibration_rows, dtype_rows, memory_rows, kv_rows]),
          "Six deterministic CSV tables generated.")

    write_csv(EVIDENCE / "weight-quantization-metrics.csv", weight_rows)
    write_csv(EVIDENCE / "linear-output-drift.csv", output_rows)
    write_csv(EVIDENCE / "activation-calibration.csv", calibration_rows)
    write_csv(EVIDENCE / "float-format-rounding.csv", dtype_rows)
    write_csv(EVIDENCE / "model-memory-estimates.csv", memory_rows)
    write_csv(EVIDENCE / "kv-cache-estimates.csv", kv_rows)

    report = {
        "lab": "LLM Engineering 06 — Quantization fundamentals",
        "date": "2026-11-14",
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "dependencies": "Python standard library only", "seed": SEED,
                        "execution": "CPU numerical emulation"},
        "scope": {"weight_matrix_shape_out_in": [rows, cols], "activation_vectors": len(activations),
                  "algorithm_implemented": "linear Round-to-Nearest (RTN) educational baseline",
                  "implemented_granularities": ["per-tensor", "per-output-channel", "per-group"],
                  "implemented_integer_formats": ["symmetric INT8", "symmetric INT4", "asymmetric affine INT4"],
                  "floating_point_emulation": ["IEEE binary16 via struct", "bfloat16 rounding from float32"],
                  "not_implemented": ["GPTQ", "AWQ", "SmoothQuant quantization recipe", "FP8 kernels",
                                      "real LLM loading", "GPU execution"]},
        "summary": {"checks_passed": sum(x["status"] == "PASS" for x in checks), "checks_total": len(checks),
                    "weight_schemes": len(weight_rows), "linear_paths": len(output_rows),
                    "smooth_scaling_transform_max_abs_error": transform_score["max_abs_error"]},
        "weight_metrics": weight_rows, "linear_output_metrics": output_rows,
        "activation_calibration_metrics": calibration_rows, "float_format_samples": dtype_rows,
        "model_memory_arithmetic": memory_rows, "kv_cache_arithmetic": kv_rows, "checks": checks,
        "interpretation": [
            "Per-channel and groupwise scales can reduce error on this deliberately heterogeneous synthetic matrix; this is not a universal ranking.",
            "Lower weight MSE does not prove lower perplexity or better task accuracy for a real LLM.",
            "Scale overhead assumes one FP32 scale per region and excludes zero-points and packing.",
            "Synthetic tensors are not pretrained model weights or real production activations.",
            "No latency, throughput, GPU allocation, kernel selection or tokens-per-second measurements are made.",
        ],
    }
    (REPORTS / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Lab report — LLM quantization fundamentals", "",
        "## Reproducibility",
        "- Date: 2026-11-14", f"- Python: {platform.python_version()}",
        "- Dependencies: standard library only; no pip install required",
        f"- Seed: {SEED}", f"- Checks: {report['summary']['checks_passed']}/{report['summary']['checks_total']} PASS",
        "- Execution: CPU numerical emulation; no GPU or CUDA", "",
        "## What was tested",
        f"- Synthetic weight matrix: {rows} output channels × {cols} input features",
        f"- Synthetic activation vectors: {len(activations)}",
        "- Linear Round-to-Nearest baseline: symmetric INT8/INT4 and asymmetric affine INT4",
        "- Scale granularities: per-tensor, per-output-channel and per-group (16/32 values)",
        "- Floating-point rounding: IEEE binary16 and bfloat16",
        "- Activation calibration: absmax vs 99th-percentile clipping",
        "- Linear projection drift: W4A16-style and W8A8-style educational paths",
        "- Algebraic channel rescaling: W' = W D and x' = D^-1 x", "",
        "## Weight reconstruction results", "",
        "| Scheme | MSE | RMSE | Max abs error | SQNR (dB) | Effective bits/weight estimate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in weight_rows:
        lines.append(f"| {row['scheme']} | {row['mse']:.8g} | {row['rmse']:.8g} | {row['max_abs_error']:.8g} | {row['sqnr_db']:.4f} | {row['effective_bits_per_weight_estimate']:.4f} |")
    lines += [
        "",
        "Effective bits include nominal integer payload plus one FP32 scale per quantization region. Zero points, packing alignment, other metadata, unquantized tensors and runtime buffers are excluded.",
        "", "## Linear projection drift", "",
        "| Path | RMSE | Max abs error | Relative L2 |",
        "|---|---:|---:|---:|",
    ]
    for row in output_rows:
        lines.append(f"| {row['path']} | {row['rmse']:.8g} | {row['max_abs_error']:.8g} | {row['relative_l2']:.8g} |")
    lines += [
        "", "Activations remain FP32 in the W4A16-style path. The W8A8-style path quantizes each input vector independently to symmetric INT8. These labels explain a precision policy; they do not reproduce a vendor kernel or a checkpoint format.",
        "", "## Activation calibration", "",
        "| Strategy | Threshold | Scale | Clipped values | RMSE | Max abs error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in calibration_rows:
        lines.append(f"| {row['strategy']} | {row['threshold']:.8g} | {row['scale']:.8g} | {row['clipped_values']} | {row['rmse']:.8g} | {row['max_abs_error']:.8g} |")
    lines += ["", "Clipping trades saturation for finer resolution over the retained range. A production threshold must be selected using representative calibration data and downstream validation.",
              "", "## Floating-point format probes", "",
              "| Input | FP16 emulation | FP16 finite | BF16 emulation | BF16 finite |",
              "|---:|---:|:---:|---:|:---:|"]
    for row in dtype_rows:
        lines.append(f"| {row['input_fp32']:.8g} | {row['fp16_emulated']} | {row['fp16_finite']} | {row['bf16_emulated']} | {row['bf16_finite']} |")
    lines += [
        "", "FP16 has more significand precision than BF16 at a similar bit width; BF16 has a much wider exponent range. This is a format conversion emulation, not a GPU throughput test.",
        "", "## Model weight memory arithmetic", "",
        "| Representation | Bits/parameter assumption | 7B parameters (GB decimal) | 7B parameters (GiB binary) |",
        "|---|---:|---:|---:|"]
    for row in memory_rows:
        lines.append(f"| {row['representation']} | {row['nominal_or_effective_bits_per_parameter']} | {row['decimal_GB']:.4f} | {row['binary_GiB']:.4f} |")
    lines += ["", "Ideal payload arithmetic. It is not a GPU memory measurement. Only the explicitly labelled INT4+scale row includes a scale overhead estimate.",
              "", "## KV cache arithmetic", "",
              "| Representation | Ideal payload (GiB) |", "|---|---:|"]
    for row in kv_rows:
        lines.append(f"| {row['representation']} | {row['ideal_payload_GiB']:.4f} |")
    lines += [
        "", f"Scenario: {layers} layers, batch {batch}, {tokens} tokens/sequence, {kv_heads} KV heads, head dimension {head_dim}.",
        "Formula: 2 × layers × batch × tokens × kv_heads × head_dim × bytes_per_element. Scale metadata and implementation buffers are excluded.",
        "", "## Checks", ""]
    for item in checks:
        lines.append(f"- [{'x' if item['status'] == 'PASS' else ' '}] **{item['status']}** — {item['check']}: {item['detail']}")
    lines += [
        "", "## Interpretation and limits", "",
        "- Inputs are deterministic synthetic values, not pretrained weights or real activations.",
        "- Only an educational linear RTN quantizer is implemented. GPTQ, AWQ and SmoothQuant are studied in the article, not reimplemented in this lab.",
        "- Per-channel/per-group rankings depend on the generated tensor distribution and are not universal.",
        "- Reconstruction error and SQNR do not replace perplexity, task evaluation, generation regression tests or human review.",
        "- The lab does not measure latency, throughput, tokens/s, GPU memory, model quality or hardware acceleration.",
        "- FP8 is discussed in the article; no FP8 kernel or GEMM implementation is claimed.",
        "", "## Evidence files", "",
        "- evidence/weight-quantization-metrics.csv",
        "- evidence/linear-output-drift.csv",
        "- evidence/activation-calibration.csv",
        "- evidence/float-format-rounding.csv",
        "- evidence/model-memory-estimates.csv",
        "- evidence/kv-cache-estimates.csv",
        "- reports/metrics.json",
    ]
    (REPORTS / "lab-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Lab: LLM Engineering 06 — Quantization fundamentals")
    print(f"Python: {platform.python_version()} | seed={SEED} | standard library only")
    print(f"Weight matrix: {rows}x{cols} | activation vectors: {len(activations)}")
    print("\nWeight reconstruction:")
    for row in weight_rows:
        print(f"  {row['scheme']:<38} RMSE={row['rmse']:.7f}  SQNR={row['sqnr_db']:.3f} dB")
    print("\nLinear projection:")
    for row in output_rows:
        print(f"  {row['path']:<76} RMSE={row['rmse']:.7f}")
    print("\nActivation calibration:")
    for row in calibration_rows:
        print(f"  {row['strategy']:<18} clipped={row['clipped_values']:>3}  RMSE={row['rmse']:.7f}")
    print(f"\nEquivalent channel transform max abs output drift: {transform_score['max_abs_error']:.3e}")
    print(f"Checks: {report['summary']['checks_passed']}/{report['summary']['checks_total']} PASS")
    for item in checks:
        print(f"  {item['status']}: {item['check']}")
    print(f"\nEvidence: {EVIDENCE}\nReports: {REPORTS}")
    return 0 if all(item["status"] == "PASS" for item in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
