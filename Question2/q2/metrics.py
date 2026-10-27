"""Loss and the E(5) / E(2) scores."""

from __future__ import annotations

import torch


def per_frame_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE over channels and pixels. pred/target are (B, H, C, H, W) -> (B, H)."""
    return (pred - target).pow(2).mean(dim=(2, 3, 4))


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over pixels and available frames of each video, then over the batch.

    Videos with two target frames and videos with five contribute the same.
    """
    frame = per_frame_mse(pred, target)
    weights = mask.to(dtype=frame.dtype)
    per_video = (frame * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
    return per_video.mean()


def horizon_means(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE at each forecast horizon, averaged over videos that have that horizon.

    Returns a length-5 tensor. Horizons with no valid videos are NaN.
    """
    frame = per_frame_mse(pred, target)
    if mask is None:
        return frame.mean(dim=0)
    means = []
    for h in range(frame.shape[1]):
        valid = mask[:, h]
        if bool(valid.any()):
            means.append(frame[valid, h].mean())
        else:
            means.append(frame.new_tensor(float("nan")))
    return torch.stack(means)


def e_score(horizon_mse: torch.Tensor, n: int) -> torch.Tensor:
    """E(n) = mean of the first n horizon MSEs."""
    return horizon_mse[:n].mean()
