"""Figures for the Question 2 report."""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .geometry import apply_group


def _gray(frame: torch.Tensor) -> np.ndarray:
    return frame.detach().cpu().float().numpy().squeeze()


def _panel(rows: list[list[torch.Tensor]], title: str, path: Path) -> None:
    n_rows = len(rows)
    n_cols = max(len(row) for row in rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.35 * n_cols, 1.45 * n_rows), squeeze=False)
    for r, row in enumerate(rows):
        for c in range(n_cols):
            axes[r, c].axis("off")
            if c < len(row) and row[c] is not None:
                axes[r, c].imshow(_gray(row[c]), cmap="gray", vmin=0, vmax=1)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _render_rows(rows: list[list[torch.Tensor]], title: str) -> Image.Image:
    n_rows = len(rows)
    n_cols = max(len(row) for row in rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.35 * n_cols, 1.45 * n_rows), squeeze=False)
    for r, row in enumerate(rows):
        for c in range(n_cols):
            axes[r, c].axis("off")
            if c < len(row) and row[c] is not None:
                axes[r, c].imshow(_gray(row[c]), cmap="gray", vmin=0, vmax=1)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def forecast_rows(
    inputs: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    mirror: bool,
    turns: int,
    pred_transformed: torch.Tensor,
) -> list[list[torch.Tensor]]:
    """Four rows of ten frames for one transformed forecast.

    inputs, prediction, target, pred_transformed are single-video tensors
    shaped (5, 1, 64, 64). prediction is f(X). pred_transformed is f(T(X)).
    """
    transformed_in = apply_group(inputs, mirror, turns)
    transformed_y = apply_group(target, mirror, turns)
    transformed_pred = apply_group(prediction, mirror, turns)
    def frames(seq: torch.Tensor) -> list[torch.Tensor]:
        return [seq[i] for i in range(seq.shape[0])]
    return [
        frames(inputs) + frames(prediction),
        frames(transformed_in) + frames(transformed_y),
        frames(transformed_in) + frames(pred_transformed),
        frames(transformed_in) + frames(transformed_pred),
    ]


def save_static_panel(rows: list[list[torch.Tensor]], title: str, path: Path) -> None:
    _panel(rows, title, path)


def save_horizon_gif(rows: list[list[torch.Tensor]], title: str, path: Path) -> None:
    """Animate the five forecast horizons. Inputs in columns 1–5 stay put."""
    frames = []
    for horizon in range(5):
        stepped = []
        for row in rows:
            future = [None] * 5
            future[horizon] = row[5 + horizon]
            stepped.append(row[:5] + future)
        frames.append(_render_rows(stepped, f"{title}  horizon {horizon + 1}"))
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=450, loop=0)


def save_figure3(panels: list[tuple[str, list[list[torch.Tensor]]]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [_render_rows(rows, title) for title, rows in panels]
    width = max(image.width for image in images)
    height = sum(image.height for image in images)
    sheet = Image.new("RGB", (width, height), "white")
    y = 0
    for image in images:
        sheet.paste(image, (0, y))
        y += image.height
    sheet.save(path)


def save_figure4(
    gap_m: np.ndarray,
    gap_r: np.ndarray,
    mark_m: list[tuple[int, float]],
    mark_r: list[tuple[int, float]],
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    axes[0].hist(gap_m, bins=30, color="#3c6e8f", weights=np.ones_like(gap_m) / len(gap_m))
    axes[1].hist(gap_r, bins=30, color="#3c6e8f", weights=np.ones_like(gap_r) / len(gap_r))
    for video_id, value in mark_m:
        axes[0].axvline(value, color="#b3452d", linewidth=1.1)
        axes[0].text(value, axes[0].get_ylim()[1] * 0.9, str(video_id), rotation=90, fontsize=8, color="#b3452d")
    for video_id, value in mark_r:
        axes[1].axvline(value, color="#b3452d", linewidth=1.1)
        axes[1].text(value, axes[1].get_ylim()[1] * 0.9, str(video_id), rotation=90, fontsize=8, color="#b3452d")
    axes[0].set_title(r"(a) per-video $C_M$")
    axes[1].set_title(r"(b) per-video $C_R$")
    for axis in axes:
        axis.set_xlabel("Consistency gap")
        axis.set_ylabel("Fraction of test videos")
    fig.suptitle("Figure 4. Seed-0 model A consistency gaps on the test set. Marked videos are those in Figure 3.", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_figure5(blocks: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]], path: Path) -> None:
    """blocks: (label, inputs (5,1,H,W), pred (2,1,H,W), truth (2,1,H,W))."""
    fig, axes = plt.subplots(len(blocks), 9, figsize=(12.5, 1.6 * len(blocks)), squeeze=False)
    for r, (label, inputs, pred, truth) in enumerate(blocks):
        shown = [inputs[i] for i in range(5)] + [pred[0], pred[1], truth[0], truth[1]]
        for c, frame in enumerate(shown):
            axes[r, c].imshow(_gray(frame), cmap="gray", vmin=0, vmax=1)
            axes[r, c].axis("off")
        axes[r, 0].set_ylabel(label, rotation=0, labelpad=28, va="center")
    fig.suptitle(
        "Figure 5. Five input frames, the first two predictions, and the first two true futures at 1×, 2× and 3×.",
        fontsize=10,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_figure2(videos: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]], path: Path) -> None:
    """Each item is (dataset index, input, prediction, target), all (5, 1, 64, 64)."""
    n = len(videos)
    fig, axes = plt.subplots(n * 3, 5, figsize=(8.5, 1.35 * n * 3), squeeze=False)
    labels = ("input", "prediction", "ground truth")
    for v, (index, inputs, pred, target) in enumerate(videos):
        sequences = (inputs, pred, target)
        for row, (label, seq) in enumerate(zip(labels, sequences)):
            for c in range(5):
                ax = axes[v * 3 + row, c]
                ax.imshow(_gray(seq[c]), cmap="gray", vmin=0, vmax=1)
                ax.axis("off")
            axes[v * 3 + row, 0].set_ylabel(f"{index}\n{label}", rotation=0, labelpad=32, va="center", fontsize=8)
    fig.suptitle("Figure 2. Seed-0 model A on three test videos. Rows are input, prediction, and ground truth.", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
