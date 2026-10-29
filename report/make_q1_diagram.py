"""Architecture diagram for the river-forecasting proposal (Lectures 4–6)."""

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
        linespacing=1.2,
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
    fig, ax = plt.subplots(figsize=(11.2, 7.6), dpi=160)
    ax.set_xlim(0, 112)
    ax.set_ylim(0, 76)
    ax.axis("off")
    ax.set_aspect("equal")
    fig.patch.set_facecolor("white")

    ax.text(
        56,
        74.0,
        "River forecast issued at time τ: water level at S1–S4 one hour ahead",
        ha="center",
        va="center",
        fontsize=11,
        color=INK,
        fontweight="medium",
    )

    box(ax, 2, 60.5, 25, 11.2, "Levels S1–S4, T1\nevery 10 min\nobs time + arrival time\n(B, 36, 5)", BLUE, 7.6)
    box(ax, 29, 60.5, 25, 11.2, "Rain R1, R2\nnative timestamps\n+ gauge locations\n(B, 36, 2)", BLUE, 7.6)
    box(
        ax,
        56,
        60.5,
        28,
        11.2,
        "Reservoir releases\nactual: effective + arrival\nschedule: announce + effective\n(B, 36, 1), (B, 6, 1)",
        BLUE,
        7.2,
    )
    box(ax, 86, 60.5, 24, 11.2, "Static river\nconnectivity\nalong-river distances\n(N, N) biases", SAND, 7.6)

    for x in (14.5, 41.5, 70, 98):
        arrow(ax, x, 60.5, x, 57.2)

    box(
        ax,
        2,
        45.5,
        108,
        11.4,
        "Timestamp alignment at issuance τ  (shared 10-min bin end-times)\n"
        "History bins τ−35 … τ (6 h). Horizon bins τ+1 … τ+6 (1 h).\n"
        "Keep a value only if obs/effective time is in the bin AND it is available by τ\n"
        "(arrival ≤ τ for levels/rain/actual; announcement ≤ τ for future schedule).\n"
        "Rain summed into bins from irregular timestamps. Project each series to features + mask + staleness + clock.",
        GREY,
        7.5,
    )
    arrow(ax, 56, 45.5, 56, 42.5)

    box(
        ax,
        2,
        31.0,
        54,
        11.0,
        "Temporal block — shared LSTM\nover 36 history steps per series\n→ embeddings (B, 8, 64) at τ\n+ future-release token (B, 1, 64)\nfrom announced schedule (B, 6, 1)",
        TEAL,
        7.5,
    )
    box(
        ax,
        58,
        31.0,
        52,
        11.0,
        "Spatial block — self-attention\n2 encoder blocks on N=9 tokens (B, 9, 64)\nmask missing → score −∞\n+ distance / upstream logit biases\n(temporal then spatial)",
        TEAL,
        7.5,
    )
    arrow(ax, 29, 31.0, 29, 28.0)
    arrow(ax, 84, 31.0, 84, 28.0)
    arrow(ax, 56, 36.5, 58, 36.5)

    box(
        ax,
        2,
        16.2,
        70,
        11.2,
        "Combination and readout\nQueries for each station × lead: q(s,k) from mixed token + lead embedding.\n"
        "Ŷ (B, 4, 6) = levels at S1–S4 for +10 … +60 min.\n"
        "No future levels or future rain as inputs. Only announced release looks ahead.",
        TEAL,
        7.5,
    )
    box(
        ax,
        74,
        16.2,
        36,
        11.2,
        "Issued product\n(B, 4)\nS1–S4 at τ+60 min\n(= lead k=6 slice)",
        GREEN,
        8,
    )
    arrow(ax, 72, 21.8, 74, 21.8)
    arrow(ax, 37, 16.2, 37, 13.2)

    box(
        ax,
        2,
        2.2,
        108,
        10.6,
        "Training: MSE on Ŷ vs archived future levels (never used as inputs). Batch B = issuances.\n"
        "Online: one forward pass every 10 min after packets with arrival ≤ τ are assembled.\n"
        "Series tokens: S1, S2, S3, S4, T1, R1, R2, reservoir history, future-release.",
        SAND,
        7.5,
    )

    fig.tight_layout(pad=0.3)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
