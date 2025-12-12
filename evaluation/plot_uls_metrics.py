import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_evaluation_rows(csv_path: Path) -> List[Tuple[str, float, float, float, float]]:
    rows: List[Tuple[str, float, float, float, float]] = []
    with csv_path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("scope") != "evaluation":
                continue
            lt = str(row.get("lesion_type", ""))
            d_mean = float(row.get("dsc_mean", 0) or 0)
            d_std = float(row.get("dsc_std", 0) or 0)
            b_mean = float(row.get("biou_mean", 0) or 0)
            b_std = float(row.get("biou_std", 0) or 0)
            rows.append((lt, d_mean, d_std, b_mean, b_std))
    return rows


def order_types(types: List[str]) -> List[str]:
    items = sorted([t for t in types if t != "ALL"])
    if "ALL" in types:
        items.append("ALL")
    return items


def plot_bars(
    labels: List[str],
    means: List[float],
    stds: List[float],
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    width = max(8.0, 0.7 * len(labels) + 2.0)
    fig, ax = plt.subplots(figsize=(width, 4.5), dpi=150)
    x = list(range(len(labels)))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color="#4C78A8", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    for rect, val in zip(bars, means):
        ax.text(rect.get_x() + rect.get_width() / 2.0, val + 0.02, f"{val:.2f}",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--outdir", type=Path, default=None)
    args = p.parse_args()

    csv_path = args.csv
    out_dir = args.outdir if args.outdir is not None else csv_path.parent

    eval_rows = load_evaluation_rows(csv_path)
    if not eval_rows:
        raise SystemExit("No evaluation rows found in CSV.")

    by_type = {t: (dmean, dstd, bmean, bstd) for t, dmean, dstd, bmean, bstd in eval_rows}
    types = order_types(list(by_type.keys()))

    d_means = [by_type[t][0] for t in types]
    d_stds = [by_type[t][1] for t in types]
    b_means = [by_type[t][2] for t in types]
    b_stds = [by_type[t][3] for t in types]

    plot_bars(types, d_means, d_stds, ylabel="Dice", title="Dice by lesion type",
              out_path=out_dir / "uls_dice_by_type.png")
    plot_bars(types, b_means, b_stds, ylabel="Boundary IoU", title="Boundary IoU by lesion type",
              out_path=out_dir / "uls_biou_by_type.png")


if __name__ == "__main__":
    main()


