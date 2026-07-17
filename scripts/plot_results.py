#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path


METHOD_ORDER = ["SP", "SFU", "Tensor Core", "Sparse Tensor Core PTX", "cuSPARSE"]
METHOD_COLORS = {
    "SP": "#777777",
    "SFU": "#4c78a8",
    "Tensor Core": "#f58518",
    "Sparse Tensor Core PTX": "#b279a2",
    "cuSPARSE": "#54a24b",
}


def number(row, key):
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return math.nan


def grouped_plot(plt, rows, cases, key, ylabel, output):
    width = 0.16
    x = list(range(len(cases)))
    plt.figure(figsize=(max(8.0, 1.8 * len(cases)), 5.5))
    for method_index, method in enumerate(METHOD_ORDER):
        by_case = {row["case_label"]: row for row in rows if row["method"] == method}
        values = [number(by_case.get(case, {}), key) for case in cases]
        offset = (method_index - (len(METHOD_ORDER) - 1) / 2) * width
        plt.bar([position + offset for position in x], values, width,
                label=method, color=METHOD_COLORS[method])
    plt.xticks(x, cases)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def speedup_plot(plt, rows, cases, reference_method, key, output):
    references = {
        row["case_label"]: number(row, key)
        for row in rows if row["method"] == reference_method
    }
    speedup_rows = []
    for row in rows:
        reference = references.get(row["case_label"], math.nan)
        latency = number(row, key)
        copied = dict(row)
        copied["speedup"] = (
            reference / latency
            if math.isfinite(reference) and math.isfinite(latency) and latency > 0
            else math.nan
        )
        speedup_rows.append(copied)
    grouped_plot(
        plt, speedup_rows, cases, "speedup",
        f"Kernel speedup relative to {reference_method}", output,
    )


def section_plot(plt, rows, cases, output):
    components = [
        ("scalar_ms", "scalar"),
        ("fused_kernel_ms", "fused WMMA"),
        ("build_power_ms", "build power"),
        ("matmul_ms", "matmul"),
        ("postprocess_ms", "postprocess"),
    ]
    labels = []
    selected = []
    for case in cases:
        for method in METHOD_ORDER:
            match = next((row for row in rows
                          if row["case_label"] == case and row["method"] == method), None)
            if match:
                selected.append(match)
                labels.append(f"{case}\n{method}")
    if not selected:
        return
    x = list(range(len(selected)))
    bottom = [0.0] * len(selected)
    plt.figure(figsize=(max(10.0, 0.65 * len(selected)), 6.0))
    for key, label in components:
        values = [number(row, key) for row in selected]
        values = [value if math.isfinite(value) else 0.0 for value in values]
        plt.bar(x, values, bottom=bottom, label=label)
        bottom = [old + value for old, value in zip(bottom, values)]
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("Separate profile-pass time (ms)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot TACT benchmark CSV output")
    parser.add_argument("csv", nargs="?", default="results/benchmark_results.csv")
    parser.add_argument("--outdir", default="results/plots")
    args = parser.parse_args()

    with open(args.csv, newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    if not all_rows:
        raise SystemExit("benchmark CSV has no data rows")

    cases = []
    for row in all_rows:
        case = row.get("case_label", row.get("functions", "unknown"))
        row.setdefault("case_label", case)
        row.setdefault("method", row.get("engine", "unknown"))
        if case not in cases:
            cases.append(case)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    skipped = [row for row in all_rows if row.get("status", "ok") != "ok"]
    with open(outdir / "status_summary.txt", "w", encoding="utf-8") as handle:
        for row in skipped:
            handle.write(
                f"{row.get('case_label', '?')} / {row.get('method', '?')}: "
                f"{row.get('status', '?')} ({row.get('log_file', '')})\n"
            )

    valid_rows = [row for row in all_rows if row.get("status", "ok") == "ok"]
    if not valid_rows:
        print(f"No successful rows to plot; wrote status summary to {outdir}")
        return

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit("plotting requires matplotlib (for example: pip install matplotlib)") from exc

    grouped_plot(plt, valid_rows, cases, "avg_ms", "Average total latency (ms)",
                 outdir / "avg_ms.png")
    grouped_plot(plt, valid_rows, cases, "kernel_ms", "Average kernel latency (ms)",
                 outdir / "kernel_ms.png")
    grouped_plot(plt, valid_rows, cases, "throughput_mvalues_s", "Throughput (Mvalues/s)",
                 outdir / "throughput.png")
    speedup_plot(plt, valid_rows, cases, "SFU", "kernel_ms",
                 outdir / "speedup_vs_sfu.png")
    speedup_plot(plt, valid_rows, cases, "SP", "kernel_ms",
                 outdir / "speedup_vs_sp.png")
    section_plot(plt, valid_rows, cases, outdir / "section_times.png")

    print(f"Wrote plots to {outdir}; skipped {len(skipped)} non-success rows")


if __name__ == "__main__":
    main()
