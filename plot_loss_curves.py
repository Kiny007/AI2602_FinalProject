"""Plot smoothed GAN loss curves from TensorBoard-exported CSV files.

Supports separate y-axis ranges for generator and discriminator so that
outliers do not compress the main trend.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot smoothed GAN loss curves from CSV files.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing the exported CSV files.")
    parser.add_argument("--g-file", type=str, default="run-tensorboard-tag-Loss_loss_g.csv", help="Generator loss CSV filename.")
    parser.add_argument("--d-file", type=str, default="run-tensorboard-tag-Loss_loss_d.csv", help="Discriminator loss CSV filename.")
    parser.add_argument("--output", type=str, default="loss_curves_smoothed_ranged.pdf", help="Output PDF filename.")
    parser.add_argument("--smoothing-span", type=int, default=0, help="EMA span. <=0 uses an automatic value.")
    parser.add_argument("--g-ymin", type=float, default=None, help="Lower bound of the generator y-axis.")
    parser.add_argument("--g-ymax", type=float, default=None, help="Upper bound of the generator y-axis.")
    parser.add_argument("--d-ymin", type=float, default=None, help="Lower bound of the discriminator y-axis.")
    parser.add_argument("--d-ymax", type=float, default=None, help="Upper bound of the discriminator y-axis.")
    parser.add_argument("--hide-raw", action="store_true", help="Hide faint raw curves and only show smoothed lines.")
    return parser.parse_args()


def load_loss_csv(csv_path: Path, value_name: str) -> pd.DataFrame:
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


def validate_range(name: str, ymin: float | None, ymax: float | None) -> None:
    if ymin is not None and ymax is not None and ymin >= ymax:
        raise ValueError(f"{name} axis range is invalid: ymin ({ymin}) must be smaller than ymax ({ymax}).")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    g_path = input_dir / args.g_file
    d_path = input_dir / args.d_file
    output_path = input_dir / args.output

    g_df = load_loss_csv(g_path, "loss_g")
    d_df = load_loss_csv(d_path, "loss_d")
    df = pd.merge(g_df, d_df, on="Step", how="inner").sort_values("Step").drop_duplicates(subset=["Step"])
    if df.empty:
        raise RuntimeError("No overlapping steps were found between generator and discriminator CSV files.")

    span = resolve_span(len(df), args.smoothing_span)
    df["loss_g_smooth"] = df["loss_g"].ewm(span=span, adjust=False).mean()
    df["loss_d_smooth"] = df["loss_d"].ewm(span=span, adjust=False).mean()

    validate_range("Generator", args.g_ymin, args.g_ymax)
    validate_range("Discriminator", args.d_ymin, args.d_ymax)

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

    fig, ax_g = plt.subplots()
    ax_d = ax_g.twinx()

    color_g = "#1f77b4"
    color_d = "#d62728"

    lines = []
    if not args.hide_raw:
        line_g_raw, = ax_g.plot(
            df["Step"], df["loss_g"], color=color_g, alpha=0.18, linewidth=1.0, label="G raw"
        )
        line_d_raw, = ax_d.plot(
            df["Step"], df["loss_d"], color=color_d, alpha=0.18, linewidth=1.0, label="D raw"
        )
        lines.extend([line_g_raw, line_d_raw])

    line_g_s, = ax_g.plot(
        df["Step"], df["loss_g_smooth"], color=color_g, linewidth=2.2, label=f"G smooth (EMA span={span})"
    )
    line_d_s, = ax_d.plot(
        df["Step"], df["loss_d_smooth"], color=color_d, linewidth=2.2, label=f"D smooth (EMA span={span})"
    )
    lines.extend([line_g_s, line_d_s])

    ax_g.set_xlabel("Training Step")
    ax_g.set_ylabel("Generator Loss", color=color_g)
    ax_d.set_ylabel("Discriminator Loss", color=color_d)
    ax_g.tick_params(axis="y", labelcolor=color_g)
    ax_d.tick_params(axis="y", labelcolor=color_d)
    ax_g.set_title("DCGAN Training Loss Curves")

    if args.g_ymin is not None or args.g_ymax is not None:
        ax_g.set_ylim(bottom=args.g_ymin, top=args.g_ymax)
    if args.d_ymin is not None or args.d_ymax is not None:
        ax_d.set_ylim(bottom=args.d_ymin, top=args.d_ymax)

    labels = [line.get_label() for line in lines]
    ax_g.legend(lines, labels, loc="upper center", ncol=2, frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved plot to: {output_path}")
    print(f"Rows: {len(df)}, smoothing span: {span}")
    if args.g_ymin is not None or args.g_ymax is not None:
        print(f"Generator axis range: [{args.g_ymin}, {args.g_ymax}]")
    if args.d_ymin is not None or args.d_ymax is not None:
        print(f"Discriminator axis range: [{args.d_ymin}, {args.d_ymax}]")


if __name__ == "__main__":
    main()
