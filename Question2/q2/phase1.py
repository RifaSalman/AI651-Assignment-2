"""Train model A and run the baseline tests. Does not train models B–D."""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch

from .checks import main as run_checks
from .data import ROOT, TEST_IDS, fixed_window, load_videos
from .evaluate import (
    blank,
    consistency_gap,
    copy_last,
    per_video_horizon_mse,
    summarise,
    time_inference,
)
from .flow import plot_figure1, pooled_flow, summarise as summarise_flow
from .geometry import apply_group
from .model import build_model, trainable_parameters
from .plots import (
    forecast_rows,
    save_figure2,
    save_figure3,
    save_figure4,
    save_figure5,
    save_horizon_gif,
    save_static_panel,
)
from .train import resolve_device, train


def output_root() -> Path:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/working").is_dir():
        if Path("/kaggle/working").is_dir() and os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
            return Path("/kaggle/working")
    return ROOT


def _device_predict(model: torch.nn.Module, device: torch.device):
    model.eval()

    @torch.no_grad()
    def predict(frames: torch.Tensor) -> torch.Tensor:
        return model(frames.to(device)).cpu()

    return predict


def _components(frame: np.ndarray) -> int:
    count, _ = cv2.connectedComponents((frame > 51).astype(np.uint8))
    return int(count - 1)


def _choose_figure2(videos: torch.Tensor) -> list[int]:
    overlap = None
    clean: list[int] = []
    for index in TEST_IDS:
        counts = [_components(videos[index, t, 0].numpy()) for t in range(10)]
        if overlap is None and min(counts) <= 1 and max(counts) >= 2:
            overlap = index
        elif len(clean) < 2 and counts and min(counts) >= 2:
            clean.append(index)
        if overlap is not None and len(clean) == 2:
            break
    if overlap is None or len(clean) < 2:
        raise RuntimeError(f"could not choose Figure 2 videos (overlap={overlap}, clean={clean})")
    return [overlap, *clean]


def _no_overlap(videos: torch.Tensor, index: int, frames: list[int]) -> bool:
    return all(_components(videos[index, frame - 1, 0].numpy()) >= 2 for frame in frames)


def _choose_figure5(videos: torch.Tensor) -> int:
    shown = {1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 13, 16, 19}
    for index in TEST_IDS:
        if _no_overlap(videos, index, sorted(shown)):
            return index
    raise RuntimeError("no test video keeps the displayed frames separated")


def _pair_report(values: list[float]) -> dict:
    return {
        "seed0": values[0],
        "seed1": values[1],
        "mean": float(sum(values) / len(values)),
    }


def _score_model(predict, videos, t: int = 1, k: int = 1, mirror: bool = False, turns: int = 0) -> dict:
    frame_mse, mask = per_video_horizon_mse(
        predict, videos, TEST_IDS, t=t, k=k, mirror=mirror, turns=turns
    )
    return summarise(frame_mse, mask)


def _load(path: Path, device: torch.device) -> torch.nn.Module:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(saved["model"])
    model.load_state_dict(saved["state_dict"])
    model.to(device).eval()
    return model


def _one_video(videos, index: int, t: int, k: int):
    inputs, target, _mask = fixed_window(videos, torch.tensor([index]), t=t, k=k)
    return inputs[0], target[0]


def evaluate_baseline(videos: torch.Tensor, out: Path, device: torch.device, checkpoints: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    indices = _choose_figure2(videos)
    figure5_index = _choose_figure5(videos)
    (out / "video_ids.json").write_text(
        json.dumps({"figure2": indices, "figure5": figure5_index}, indent=2)
    )

    seed_paths = [checkpoints / f"A_seed{seed}.pt" for seed in (0, 1)]
    original = []
    mirror_error = []
    rotation_error = []
    mirror_gap_mean = []
    rotation_gap_mean = []
    speed = {1: [], 2: [], 3: []}
    inference_seconds = []
    gaps_seed0 = {}

    inputs_f2, targets_f2, _ = fixed_window(videos, torch.tensor(indices), t=1, k=1)
    figure2_preds = None

    for seed, path in enumerate(seed_paths):
        model = _load(path, device)
        predict = _device_predict(model, device)
        original.append(_score_model(predict, videos))
        mirror_error.append(_score_model(predict, videos, mirror=True))
        rotation_error.append(_score_model(predict, videos, turns=1))
        gap_m = consistency_gap(predict, videos, TEST_IDS, mirror=True, turns=0)
        gap_r = consistency_gap(predict, videos, TEST_IDS, mirror=False, turns=1)
        mirror_gap_mean.append(float(gap_m.mean()))
        rotation_gap_mean.append(float(gap_r.mean()))
        for k in (1, 2, 3):
            speed[k].append(_score_model(predict, videos, k=k))
        inference_seconds.append(time_inference(predict, videos))
        if seed == 0:
            gaps_seed0 = {"CM": gap_m.numpy(), "CR": gap_r.numpy()}
            figure2_preds = predict(inputs_f2)
            _figures_seed0(videos, predict, indices, figure5_index, inputs_f2, targets_f2, figure2_preds, gap_m, gap_r, out)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    blank_predict = blank
    copy_predict = copy_last
    reference = {
        "Blank": {
            "original": _score_model(blank_predict, videos),
            "mirror_error": _score_model(blank_predict, videos, mirror=True),
            "rotation_error": _score_model(blank_predict, videos, turns=1),
            "CM": float(consistency_gap(blank_predict, videos, TEST_IDS, True, 0).mean()),
            "CR": float(consistency_gap(blank_predict, videos, TEST_IDS, False, 1).mean()),
            "speed": {str(k): _score_model(blank_predict, videos, k=k) for k in (1, 2, 3)},
        },
        "Copy-last": {
            "original": _score_model(copy_predict, videos),
            "mirror_error": _score_model(copy_predict, videos, mirror=True),
            "rotation_error": _score_model(copy_predict, videos, turns=1),
            "CM": float(consistency_gap(copy_predict, videos, TEST_IDS, True, 0).mean()),
            "CR": float(consistency_gap(copy_predict, videos, TEST_IDS, False, 1).mean()),
            "speed": {str(k): _score_model(copy_predict, videos, k=k) for k in (1, 2, 3)},
        },
    }
    payload = {
        "parameters": trainable_parameters(build_model("A")),
        "inference_seconds": inference_seconds,
        "A": {
            "original": original,
            "EM": mirror_error,
            "ER": rotation_error,
            "CM": mirror_gap_mean,
            "CR": rotation_gap_mean,
            "speed": {str(k): speed[k] for k in speed},
        },
        "reference": reference,
    }
    (out / "baseline_scores.json").write_text(json.dumps(payload, indent=2))
    np.savez(out / "consistency_seed0.npz", CM=gaps_seed0["CM"], CR=gaps_seed0["CR"])
    print(json.dumps({"parameters": payload["parameters"], "A_E5": [row["E5"] for row in original]}, indent=2))


def _figures_seed0(videos, predict, indices, figure5_index, inputs_f2, targets_f2, preds, gap_m, gap_r, out: Path):
    save_figure2(
        [(indices[i], inputs_f2[i], preds[i], targets_f2[i]) for i in range(len(indices))],
        out / "figure2.png",
    )
    top_m = torch.argsort(gap_m, descending=True)[:2].tolist()
    top_r = torch.argsort(gap_r, descending=True)[:2].tolist()
    panels = []
    gif_dir = out / "gifs"
    for kind, positions, mirror, turns in (
        ("mirror", top_m, True, 0),
        ("rotation", top_r, False, 1),
    ):
        for rank, position in enumerate(positions):
            dataset_id = 9000 + position
            inputs, target = _one_video(videos, dataset_id, 1, 1)
            pred = predict(inputs.unsqueeze(0))[0]
            pred_t = predict(apply_group(inputs.unsqueeze(0), mirror, turns))[0]
            rows = forecast_rows(inputs, pred, target, mirror, turns, pred_t)
            title = f"{kind} video {dataset_id}"
            panels.append((title, rows))
            save_static_panel(rows, title, out / f"figure3_{kind}_{rank}.png")
            save_horizon_gif(rows, title, gif_dir / f"figure3_{kind}_{rank}.gif")
    save_figure3(panels, out / "figure3.png")
    save_figure4(
        gap_m.numpy(),
        gap_r.numpy(),
        [(9000 + i, float(gap_m[i])) for i in top_m],
        [(9000 + i, float(gap_r[i])) for i in top_r],
        out / "figure4.png",
    )
    (out / "figure3_ids.json").write_text(
        json.dumps(
            {
                "mirror": [9000 + i for i in top_m],
                "rotation": [9000 + i for i in top_r],
                "mirror_gap": [float(gap_m[i]) for i in top_m],
                "rotation_gap": [float(gap_r[i]) for i in top_r],
            },
            indent=2,
        )
    )
    blocks = []
    for k, label in ((1, "1×"), (2, "2×"), (3, "3×")):
        inputs, target = _one_video(videos, figure5_index, 1, k)
        pred = predict(inputs.unsqueeze(0))[0]
        blocks.append((label, inputs, pred[:2], target[:2]))
    save_figure5(blocks, out / "figure5.png")


def main() -> None:
    out = output_root()
    figures = out / "outputs"
    figures.mkdir(parents=True, exist_ok=True)
    device = resolve_device()
    print(f"output {out} device {device}", flush=True)
    run_checks()
    videos = load_videos()
    dx, dy = pooled_flow(videos, n_videos=50)
    stats = summarise_flow(dx, dy)
    plot_figure1(stats, figures / "figure1.png")
    (figures / "figure1_stats.json").write_text(
        json.dumps(
            {
                "n_pixels": stats["n_pixels"],
                "median_speed": stats["median_speed"],
                "direction_fraction": stats["direction_fraction"],
                "direction_bin_edges": stats["direction_bin_edges"],
                "speed_mean": float(np.mean(stats["speed"])),
                "speed_p90": float(np.percentile(stats["speed"], 90)),
            },
            indent=2,
        )
    )
    for seed in (0, 1):
        train("A", seed, device=device, output_dir=out)
    evaluate_baseline(videos, figures, device, out / "checkpoints")
    # Checkpoints were written under out/checkpoints. Copy scores next to them.
    print("phase 1 complete", flush=True)


if __name__ == "__main__":
    main()
