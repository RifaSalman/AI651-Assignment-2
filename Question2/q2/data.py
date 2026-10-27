"""Moving MNIST, loaded the way the handout specifies."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torchvision.datasets import MovingMNIST

from .geometry import GROUP, apply_group
from .windows import allowed_starts, window_frames

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"

TRAIN_IDS = range(0, 8000)
VAL_IDS = range(8000, 9000)
TEST_IDS = range(9000, 10000)

_VIDEOS: torch.Tensor | None = None


def resolve_data_root(root: Path | None = None) -> Path:
    """Directory passed to torchvision. Uses a local or Kaggle copy if one exists."""
    if root is not None:
        return Path(root)
    if (DATA_ROOT / "MovingMNIST" / "mnist_test_seq.npy").exists():
        return DATA_ROOT
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        matches = list(kaggle_input.rglob("mnist_test_seq.npy"))
        if matches:
            npy = matches[0]
            if npy.parent.name == "MovingMNIST":
                return npy.parent.parent
            link_root = Path("/kaggle/working/data")
            dest = link_root / "MovingMNIST" / "mnist_test_seq.npy"
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                dest.symlink_to(npy)
            return link_root
    # Kaggle input is read-only, so a missing download has to land in /kaggle/working.
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return Path("/kaggle/working/data")
    return DATA_ROOT


def load_videos(root: Path | None = None) -> torch.Tensor:
    """All 10,000 videos as uint8, shape (10000, 20, 1, 64, 64)."""
    global _VIDEOS
    if _VIDEOS is None:
        dataset = MovingMNIST(resolve_data_root(root), split=None, download=True)
        videos = dataset.data
        if videos.shape != (10000, 20, 1, 64, 64):
            raise RuntimeError(f"unexpected Moving MNIST shape {tuple(videos.shape)}")
        _VIDEOS = videos
    return _VIDEOS


def sample_window_specs(
    batch_size: int,
    generator: torch.Generator,
    speed_augmented: bool,
    symmetry: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw (t, k, group index) for one batch. Group index is -1 if symmetry is off."""
    if speed_augmented:
        k = torch.randint(1, 4, (batch_size,), generator=generator)
        t = torch.empty(batch_size, dtype=torch.long)
        for speed in (1, 2, 3):
            starts = allowed_starts(speed, True)
            pick = torch.randint(0, len(starts), (batch_size,), generator=generator)
            chosen = torch.tensor(starts, dtype=torch.long)[pick]
            t = torch.where(k == speed, chosen, t)
    else:
        k = torch.ones(batch_size, dtype=torch.long)
        t = torch.randint(1, 12, (batch_size,), generator=generator)
    if symmetry:
        g = torch.randint(0, len(GROUP), (batch_size,), generator=generator)
    else:
        g = torch.full((batch_size,), -1, dtype=torch.long)
    return t, k, g


def collate_windows(
    videos: torch.Tensor,
    indices: torch.Tensor,
    t: torch.Tensor,
    k: torch.Tensor,
    group: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a float batch in [0, 1].

    Returns x (B, 5, 1, 64, 64), y (B, 5, 1, 64, 64) and a mask (B, 5) that is
    True on target frames that exist. Missing targets stay zero and are masked
    out of the loss.
    """
    batch = indices.shape[0]
    x = torch.empty(batch, 5, 1, 64, 64, dtype=torch.float32)
    y = torch.zeros(batch, 5, 1, 64, 64, dtype=torch.float32)
    mask = torch.zeros(batch, 5, dtype=torch.bool)
    for i in range(batch):
        inputs, targets = window_frames(int(t[i]), int(k[i]))
        clip = videos[int(indices[i])].to(dtype=torch.float32).div(255)
        x[i] = clip[[f - 1 for f in inputs]]
        if targets:
            y[i, : len(targets)] = clip[[f - 1 for f in targets]]
            mask[i, : len(targets)] = True
        gi = int(group[i])
        if gi >= 0:
            mirror, turns = GROUP[gi]
            x[i] = apply_group(x[i], mirror, turns)
            y[i] = apply_group(y[i], mirror, turns)
    return x, y, mask


def fixed_window(
    videos: torch.Tensor,
    indices: range | torch.Tensor,
    t: int = 1,
    k: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same (t, k) for every index. Used for test evaluation."""
    if isinstance(indices, range):
        indices = torch.tensor(list(indices), dtype=torch.long)
    else:
        indices = indices.to(dtype=torch.long)
    group = torch.full((indices.shape[0],), -1, dtype=torch.long)
    tt = torch.full((indices.shape[0],), t, dtype=torch.long)
    kk = torch.full((indices.shape[0],), k, dtype=torch.long)
    return collate_windows(videos, indices, tt, kk, group)
