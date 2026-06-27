"""Plot d_real and d_fake curves with a shared y-axis from CSV files."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot d_real and d_fake curves with a shared y-axis.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing the exported CSV files.")
    parser.add_argument("--real-file", type=str, default="run-tensorboard-tag-Loss_d_real.csv", help="d_real CSV filename.")
    parser.add_argument("--fake-file", type=str, default="run-tensorboard-tag-Loss_d_fake.csv", help="d_fake CSV filename.")
    parser.add_argument("--output", type=str, default="d_real_fake_curves.pdf", help="Output PDF filename.")
    parser.add_argument("--smoothing-span", type=int, default=0, help="EMA span. <=0 uses an automatic value.")
    parser.add_argument("--ymin", type=float, default=None, help="Lower bound of the shared y-axis.")
    parser.add_argument("--ymax", type=float, default=None, help="Upper bound of the shared y-axis.")
    parser.add_argument("--hide-raw", action="store_true", help="Hide faint raw curves and only show smoothed lines.")
    return parser.parse_args()


def load_csv(csv_path: Path, value_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_columns = {"Step", "Value"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
    return df.rename(columns={"Value": value_name})[["Step", value_name]]


def resolve_span(num_rows: int, requested_span: int) -> int:
    if requested_span > 0:
        return requested_span
    span = max(5, min(101, int(math.ceil(num_rows * 0.05))))
    if span % 2 == 0:
        span += 1
    return span


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    real_path = input_dir / args.real_file
    fake_path = input_dir / args.fake_file
    output_path = input_dir / args.output

    if args.ymin is not None and args.ymax is not None and args.ymin >= args.ymax:
        raise ValueError(f"Invalid y-axis range: ymin ({args.ymin}) must be smaller than ymax ({args.ymax}).")

    real_df = load_csv(real_path, "d_real")
    fake_df = load_csv(fake_path, "d_fake")
    df = pd.merge(real_df, fake_df, on="Step", how="inner").sort_values("Step").drop_duplicates(subset=["Step"])
    if df.empty:
        raise RuntimeError("No overlapping steps were found between d_real and d_fake CSV files.")

    span = resolve_span(len(df), args.smoothing_span)
    df["d_real_smooth"] = df["d_real"].ewm(span=span, adjust=False).mean()
    df["d_fake_smooth"] = df["d_fake"].ewm(span=span, adjust=False).mean()

    plt.rcParams.update(
        {
            "figure.figsize": (11, 6),
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots()

    color_real = "#1f77b4"
    color_fake = "#d62728"
    lines = []

    if not args.hide_raw:
        line_real_raw, = ax.plot(
            df["Step"], df["d_real"], color=color_real, alpha=0.18, linewidth=1.0, label="D(real) raw"
        )
        line_fake_raw, = ax.plot(
            df["Step"], df["d_fake"], color=color_fake, alpha=0.18, linewidth=1.0, label="D(fake) raw"
        )
        lines.extend([line_real_raw, line_fake_raw])

    line_real_s, = ax.plot(
        df["Step"], df["d_real_smooth"], color=color_real, linewidth=2.2, label=f"D(real) smooth (EMA span={span})"
    )
    line_fake_s, = ax.plot(
        df["Step"], df["d_fake_smooth"], color=color_fake, linewidth=2.2, label=f"D(fake) smooth (EMA span={span})"
    )
    lines.extend([line_real_s, line_fake_s])

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Discriminator Confidence")
    ax.set_title("D(real) and D(fake) Curves")
    if args.ymin is not None or args.ymax is not None:
        ax.set_ylim(bottom=args.ymin, top=args.ymax)

    ax.legend(lines, [line.get_label() for line in lines], loc="upper center", ncol=2, frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved plot to: {output_path}")
    print(f"Rows: {len(df)}, smoothing span: {span}")
    if args.ymin is not None or args.ymax is not None:
        print(f"Shared axis range: [{args.ymin}, {args.ymax}]")


if __name__ == "__main__":
    main()
