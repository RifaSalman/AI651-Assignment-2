"""Optical flow on the first 50 training videos, for Figure 1."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import ROOT, load_videos

OUT = ROOT / "outputs"


def pooled_flow(videos: torch.Tensor, n_videos: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Farneback flow on consecutive frames. Returns dx, dy of the kept pixels."""
    dx_parts: list[np.ndarray] = []
    dy_parts: list[np.ndarray] = []
    for index in range(n_videos):
        clip = videos[index, :, 0].numpy()
        for s in range(19):
            prev = clip[s]
            nxt = clip[s + 1]
            flow = cv2.calcOpticalFlowFarneback(
                prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            keep = (prev > 51) & (nxt > 51) & (magnitude > 0.5)
            if not np.any(keep):
                continue
            dx_parts.append(flow[..., 0][keep])
            dy_parts.append(flow[..., 1][keep])
    if not dx_parts:
        raise RuntimeError("no flow pixels passed the intensity and magnitude gates")
    return np.concatenate(dx_parts), np.concatenate(dy_parts)


def summarise(dx: np.ndarray, dy: np.ndarray) -> dict:
    speed = np.sqrt(dx**2 + dy**2)
    theta = np.degrees(np.arctan2(-dy, dx)) % 360
    direction_bins = np.linspace(0, 360, 19)
    direction_counts, _ = np.histogram(theta, bins=direction_bins)
    direction_fraction = direction_counts / direction_counts.sum()
    return {
        "n_pixels": int(dx.shape[0]),
        "median_speed": float(np.median(speed)),
        "direction_fraction": direction_fraction.tolist(),
        "direction_bin_edges": direction_bins.tolist(),
        "speed": speed,
        "theta": theta,
    }


def plot_figure1(stats: dict, path: Path) -> None:
    theta = stats["theta"]
    speed = stats["speed"]
    edges = np.array(stats["direction_bin_edges"])
    fraction = np.array(stats["direction_fraction"])
    centres = 0.5 * (edges[:-1] + edges[1:])

    speed_edges = np.linspace(0, max(float(speed.max()), 1.0), 31)
    speed_counts, _ = np.histogram(speed, bins=speed_edges)
    speed_fraction = speed_counts / speed_counts.sum()
    speed_centres = 0.5 * (speed_edges[:-1] + speed_edges[1:])
    width = speed_edges[1] - speed_edges[0]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    axes[0].bar(centres, fraction, width=20, align="center", color="#3c6e8f", edgecolor="white")
    axes[0].axhline(1 / 18, color="#b3452d", linewidth=1.2, label="uniform (1/18)")
    axes[0].set_xlim(0, 360)
    axes[0].set_xticks(range(0, 361, 45))
    axes[0].set_xlabel("Flow direction (degrees)")
    axes[0].set_ylabel("Fraction of kept pixels")
    axes[0].set_title("(a) Direction")
    axes[0].legend(frameon=False)

    axes[1].bar(
        speed_centres,
        speed_fraction,
        width=width,
        align="center",
        color="#3c6e8f",
        edgecolor="white",
    )
    axes[1].axvline(stats["median_speed"], color="#b3452d", linewidth=1.2, label="median")
    axes[1].set_xlabel("Flow speed (pixels per frame)")
    axes[1].set_ylabel("Fraction of kept pixels")
    axes[1].set_title("(b) Speed")
    axes[1].legend(frameon=False)

    fig.suptitle(
        f"Figure 1. Optical flow on the first 50 training videos "
        f"({stats['n_pixels']} pixels kept, median speed {stats['median_speed']:.3g} px/frame).",
        fontsize=10,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    videos = load_videos()
    dx, dy = pooled_flow(videos, n_videos=50)
    stats = summarise(dx, dy)
    OUT.mkdir(parents=True, exist_ok=True)
    plot_figure1(stats, OUT / "figure1.png")
    payload = {
        "n_pixels": stats["n_pixels"],
        "median_speed": stats["median_speed"],
        "direction_fraction": stats["direction_fraction"],
        "direction_bin_edges": stats["direction_bin_edges"],
        "speed_mean": float(np.mean(stats["speed"])),
        "speed_p90": float(np.percentile(stats["speed"], 90)),
    }
    (OUT / "figure1_stats.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"wrote {OUT / 'figure1.png'}")


if __name__ == "__main__":
    main()
