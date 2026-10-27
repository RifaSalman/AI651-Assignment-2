"""Test-set scores, consistency gaps, and the symmetrised forecaster."""

from __future__ import annotations

import time
from collections.abc import Callable

import torch

from .data import TEST_IDS, fixed_window
from .geometry import GROUP, apply_group, group_inverse
from .metrics import per_frame_mse

PredictFn = Callable[[torch.Tensor], torch.Tensor]


def blank(frames: torch.Tensor) -> torch.Tensor:
    return frames.new_zeros(frames.shape[0], 5, 1, 64, 64)


def copy_last(frames: torch.Tensor) -> torch.Tensor:
    return frames[:, -1:].expand(-1, 5, -1, -1, -1).contiguous()


def symmetrised(model: torch.nn.Module, frames: torch.Tensor) -> torch.Tensor:
    """Average the eight group-transformed predictions. Needs no retraining."""
    total = frames.new_zeros(frames.shape[0], 5, 1, 64, 64)
    for mirror, turns in GROUP:
        transformed = apply_group(frames, mirror, turns)
        prediction = model(transformed)
        inv_mirror, inv_turns = group_inverse(mirror, turns)
        total = total + apply_group(prediction, inv_mirror, inv_turns)
    return total / len(GROUP)


@torch.no_grad()
def per_video_horizon_mse(
    predict: PredictFn,
    videos: torch.Tensor,
    indices: range | torch.Tensor,
    t: int = 1,
    k: int = 1,
    batch_size: int = 32,
    mirror: bool = False,
    turns: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-video MSE at each horizon, and the valid-frame mask.

    When mirror/turns are set, both the input and the target are transformed,
    so the score is E_T rather than the error on the original clip.
    """
    if isinstance(indices, range):
        indices = torch.tensor(list(indices), dtype=torch.long)
    frames_mse = []
    masks = []
    for start in range(0, indices.shape[0], batch_size):
        idx = indices[start : start + batch_size]
        inputs, targets, mask = fixed_window(videos, idx, t=t, k=k)
        if mirror or turns:
            inputs = apply_group(inputs, mirror, turns)
            targets = apply_group(targets, mirror, turns)
        prediction = predict(inputs)
        frames_mse.append(per_frame_mse(prediction, targets))
        masks.append(mask)
    return torch.cat(frames_mse, dim=0), torch.cat(masks, dim=0)


@torch.no_grad()
def consistency_gap(
    predict: PredictFn,
    videos: torch.Tensor,
    indices: range | torch.Tensor,
    mirror: bool,
    turns: int,
    batch_size: int = 32,
    t: int = 1,
    k: int = 1,
) -> torch.Tensor:
    """Per-video C_T at the requested speed. Shape (N,)."""
    if isinstance(indices, range):
        indices = torch.tensor(list(indices), dtype=torch.long)
    gaps = []
    for start in range(0, indices.shape[0], batch_size):
        idx = indices[start : start + batch_size]
        inputs, _targets, mask = fixed_window(videos, idx, t=t, k=k)
        original = predict(inputs)
        transformed_in = apply_group(inputs, mirror, turns)
        transformed_pred = predict(transformed_in)
        moved = apply_group(original, mirror, turns)
        # Score only horizons that exist, matching E(5) / E(2) at this window.
        frame = per_frame_mse(transformed_pred, moved)
        weights = mask.to(dtype=frame.dtype)
        gaps.append((frame * weights).sum(dim=1) / weights.sum(dim=1))
    return torch.cat(gaps, dim=0)


def summarise(frame_mse: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    means = []
    for h in range(frame_mse.shape[1]):
        valid = mask[:, h]
        means.append(float(frame_mse[valid, h].mean()) if bool(valid.any()) else float("nan"))
    out = {f"MSE{h + 1}": means[h] for h in range(5)}
    finite = [m for m in means if m == m]
    out["E5"] = float(sum(means[:5]) / 5) if all(m == m for m in means[:5]) else float("nan")
    out["E2"] = float(sum(means[:2]) / 2) if all(m == m for m in means[:2]) else float("nan")
    out["horizons_present"] = float(len(finite))
    return out


@torch.no_grad()
def time_inference(
    predict: PredictFn,
    videos: torch.Tensor,
    batch_size: int = 32,
) -> float:
    """Wall clock for the 1,000 original test videos, after each batch is ready."""
    indices = torch.tensor(list(TEST_IDS), dtype=torch.long)
    batches = []
    for start in range(0, indices.shape[0], batch_size):
        inputs, _targets, _mask = fixed_window(videos, indices[start : start + batch_size], t=1, k=1)
        batches.append(inputs)
    # One warmup batch, not included in the time.
    predict(batches[0])
    t0 = time.perf_counter()
    for inputs in batches:
        predict(inputs)
    return time.perf_counter() - t0


@torch.no_grad()
def time_model(
    model: torch.nn.Module,
    videos: torch.Tensor,
    device: torch.device,
    wrapped: bool = False,
    batch_size: int = 32,
) -> float:
    """Time the forward pass after each batch is already on device.

    The host-to-device copy is excluded. For a symmetrised forecaster the
    eight transforms, forwards, inverses and the average are included.
    """
    model.eval()
    indices = torch.tensor(list(TEST_IDS), dtype=torch.long)
    batches = []
    for start in range(0, indices.shape[0], batch_size):
        inputs, _targets, _mask = fixed_window(
            videos, indices[start : start + batch_size], t=1, k=1
        )
        batches.append(inputs.to(device))

    def run(frames: torch.Tensor) -> torch.Tensor:
        if wrapped:
            return symmetrised(model, frames)
        return model(frames)

    run(batches[0])
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for frames in batches:
        run(frames)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def load_forecaster(path, build):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model = build(saved["model"])
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model
