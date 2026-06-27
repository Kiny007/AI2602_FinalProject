"""Plot a square metric curve from a TensorBoard-exported CSV file."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a square metric curve from a CSV file.")
    parser.add_argument("--input-csv", type=str, required=True, help="Path to the metric CSV file.")
    parser.add_argument("--output", type=str, default="", help="Output PDF path. Defaults next to the CSV file.")
    parser.add_argument("--title", type=str, default="Metric Curve", help="Plot title.")
    parser.add_argument("--ylabel", type=str, default="Metric Value", help="Y-axis label.")
    parser.add_argument("--color", type=str, default="#2ca02c", help="Main line color.")
    parser.add_argument("--smoothing-span", type=int, default=0, help="EMA span. <=0 means no smoothing.")
    parser.add_argument("--ymin", type=float, default=None, help="Lower bound of the y-axis.")
    parser.add_argument("--ymax", type=float, default=None, help="Upper bound of the y-axis.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv).resolve()
    output_path = Path(args.output).resolve() if args.output else input_csv.with_suffix(".pdf")

    df = pd.read_csv(input_csv)
    required_columns = {"Step", "Value"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"{input_csv} is missing columns: {sorted(missing)}")
    if df.empty:
        raise RuntimeError(f"{input_csv} has no rows.")
    if args.ymin is not None and args.ymax is not None and args.ymin >= args.ymax:
        raise ValueError(f"Invalid y-axis range: ymin ({args.ymin}) must be smaller than ymax ({args.ymax}).")

    df = df[["Step", "Value"]].sort_values("Step").drop_duplicates(subset=["Step"])
    use_smoothing = args.smoothing_span > 0
    if use_smoothing:
        df["Value_plot"] = df["Value"].ewm(span=args.smoothing_span, adjust=False).mean()
    else:
        df["Value_plot"] = df["Value"]

    plt.rcParams.update(
        {
            "figure.figsize": (7, 7),
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots()

    line_metric, = ax.plot(
        df["Step"],
        df["Value_plot"],
        color=args.color,
        linewidth=2.2,
        label=f"EMA smooth (span={args.smoothing_span})" if use_smoothing else "Metric",
    )

    ax.set_xlabel("Training Step")
    ax.set_ylabel(args.ylabel)
    ax.set_title(args.title)
    if args.ymin is not None or args.ymax is not None:
        ax.set_ylim(bottom=args.ymin, top=args.ymax)

    ax.legend([line_metric], [line_metric.get_label()], loc="best", frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved plot to: {output_path}")
    print(f"Rows: {len(df)}, smoothing span: {args.smoothing_span if use_smoothing else 'disabled'}")


if __name__ == "__main__":
    main()
