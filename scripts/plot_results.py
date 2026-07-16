#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


METHOD_NAMES = {
    "baseline_fast": "sfu",
    "baseline_sp": "sp",
    "wmma": "wmma (tact)",
    "cusparse": "cuSparse",
    "cusparselt": "cuSparseLt",
}


def main():
    parser = argparse.ArgumentParser(description="Plot TACT benchmark CSV output")
    parser.add_argument("csv", nargs="?", default="results/benchmark_results.csv")
    parser.add_argument("--outdir", default="results/plots")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit("plotting requires matplotlib (for example: pip install matplotlib)") from exc

    with open(args.csv, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("benchmark CSV has no data rows")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    labels = [f"{METHOD_NAMES.get(r['engine'], r['engine'])}\n{r['functions']}" for r in rows]
    ms = [float(r["avg_ms"]) for r in rows]
    kernel_ms = [float(r.get("avg_kernel_ms", r["avg_ms"])) for r in rows]
    throughput = [float(r["throughput_mvalues_s"]) for r in rows]

    for values, ylabel, filename in (
        (ms, "Average HtoD + kernel + DtoH ms", "avg_ms.png"),
        (kernel_ms, "Average kernel ms", "kernel_ms.png"),
        (throughput, "Mvalues/s", "throughput.png"),
    ):
        width = max(8.0, 0.55 * len(rows))
        plt.figure(figsize=(width, 5.5))
        plt.bar(labels, values)
        plt.ylabel(ylabel)
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(outdir / filename, dpi=160)
        plt.close()

    print(f"Wrote timing and throughput plots to {outdir}")


if __name__ == "__main__":
    main()
