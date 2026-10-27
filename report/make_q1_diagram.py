"""Architecture diagram for the river-forecasting proposal."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "figures" / "q1_architecture.png"

BLUE = "#d6e6f5"
TEAL = "#d5efe8"
SAND = "#f6ecd4"
GREEN = "#e3f2d8"
GREY = "#eef0f2"
INK = "#1c2833"
MUTED = "#3d4c5c"


def box(ax, x, y, w, h, text, face, size=8.0, weight="regular"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.08",
        facecolor=face,
        edgecolor=INK,
        linewidth=0.8,
        mutation_aspect=0.6,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color=INK,
        fontweight=weight,
        linespacing=1.25,
        wrap=True,
    )


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=0.8,
            color=INK,
            shrinkA=0,
            shrinkB=0,
        )
    )


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.2, 7.4), dpi=160)
    ax.set_xlim(0, 112)
    ax.set_ylim(0, 74)
    ax.axis("off")
    ax.set_aspect("equal")
    fig.patch.set_facecolor("white")

    ax.text(
        56,
        72.2,
        "River forecast issued at time τ: water level at S1–S4 one hour ahead",
        ha="center",
        va="center",
        fontsize=11,
        color=INK,
        fontweight="medium",
    )

    # Sources
    box(ax, 2, 58.5, 24, 11, "Levels S1–S4, T1\nevery 10 min\n(B, 36, 5)", BLUE, 8)
    box(ax, 28, 58.5, 24, 11, "Rain R1, R2\ntimestamps, locations\n(B, 36, 2)", BLUE, 8)
    box(ax, 54, 58.5, 28, 11, "Reservoir releases\npast actual + schedule\nfuture schedule\n(B, 36, 1), (B, 6, 1)", BLUE, 7.6)
    box(ax, 84, 58.5, 26, 11, "Static graph\nconnectivity, distances\nedge index (2, E)\ndistance (E,)", SAND, 7.6)

    arrow(ax, 14, 58.5, 14, 55.2)
    arrow(ax, 40, 58.5, 40, 55.2)
    arrow(ax, 68, 58.5, 68, 55.2)
    arrow(ax, 97, 58.5, 97, 55.2)

    box(
        ax,
        2,
        44.2,
        108,
        10.6,
        "Alignment at issuance τ\n"
        "Shared 10-min axis. History is τ−35 … τ (36 steps, 6 h). Horizon is τ+1 … τ+6 (1 h).\n"
        "Keep a value only if it was observed and had arrived by τ. Missing or late → mask 0, value 0.\n"
        "Rain is summed into the same bins. Future rain is not filled in. Future input is the release schedule only.",
        GREY,
        8,
    )
    arrow(ax, 56, 44.2, 56, 41.2)

    box(
        ax,
        2,
        30.6,
        52,
        10.2,
        "Node features  (B, 36, 9, F)\n"
        "Nodes: Res, S1, T1, S2, U, S3, S4, R1, R2\n"
        "F = value, mask, staleness, time-of-day,\n"
        "day-of-year. U has no level.",
        TEAL,
        7.6,
    )
    box(
        ax,
        58,
        30.6,
        52,
        10.2,
        "Directed graph, downstream only\n"
        "Res→S1→S2→S3→S4\n"
        "T1→S2    U→S3\n"
        "R1→U    R2→S3 and R2→S4",
        SAND,
        7.6,
    )
    arrow(ax, 28, 30.6, 28, 27.6)
    arrow(ax, 84, 30.6, 84, 27.6)

    box(
        ax,
        2,
        16.4,
        108,
        10.8,
        "Learned blocks, shared weights\n"
        "Per-node GRU over the 36-step history. Each downstream edge sends a lag mixture:\n"
        "m(u→v, t) = Σℓ αℓ(distance, upstream level) · z(u, t−ℓ), ℓ = 0…35.\n"
        "Four message-passing steps cover reservoir → S4 and R1 → U → S3 → S4 in one update.",
        TEAL,
        8,
    )
    arrow(ax, 56, 16.4, 56, 13.4)

    box(
        ax,
        2,
        2.2,
        66,
        10.8,
        "Six-step rollout, no future levels or rain\n"
        "At step k the reservoir node receives the announced release.\n"
        "Other nodes receive only messages from upstream.\n"
        "Hidden state carries travel that is still in transit.",
        TEAL,
        7.8,
    )
    box(
        ax,
        72,
        2.2,
        38,
        10.8,
        "Readout at S1–S4\n"
        "(B, 6, 4, 3)\n"
        "quantiles 0.1, 0.5, 0.9\n"
        "Deliver k = 6, the +60 min level",
        GREEN,
        8,
    )
    arrow(ax, 68, 7.6, 72, 7.6)

    fig.tight_layout(pad=0.3)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
