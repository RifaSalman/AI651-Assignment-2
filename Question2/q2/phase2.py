"""Train B, C, D and D+B, then score the symmetrised forecasters.

Run this only after Q2.11 has been committed. Model A is not retrained.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import torch

from .data import TEST_IDS, fixed_window, load_videos
from .evaluate import (
    consistency_gap,
    per_video_horizon_mse,
    summarise,
    symmetrised,
    time_model,
)
from .geometry import apply_group
from .model import build_model, trainable_parameters
from .phase1 import _one_video, output_root
from .plots import forecast_rows, save_figure3, save_figure5, save_horizon_gif, save_static_panel
from .train import resolve_device, train

MODELS = ("B", "C", "D", "DB")


def _find(name: str) -> Path:
    direct = output_root() / "checkpoints" / name
    if direct.exists():
        return direct
    found = next(Path("/kaggle/input").rglob(name), None)
    if found is None:
        raise FileNotFoundError(name)
    return found


def _read_json(name: str) -> dict:
    direct = output_root() / "outputs" / name
    if direct.exists():
        return json.loads(direct.read_text())
    found = next(Path("/kaggle/input").rglob(name), None)
    if found is None:
        raise FileNotFoundError(name)
    return json.loads(found.read_text())


def _device_predict(model, device, wrapped: bool = False):
    model.eval()

    @torch.no_grad()
    def predict(frames: torch.Tensor) -> torch.Tensor:
        frames = frames.to(device)
        output = symmetrised(model, frames) if wrapped else model(frames)
        return output.cpu()

    return predict


def _load(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(saved["model"])
    model.load_state_dict(saved["state_dict"])
    model.to(device).eval()
    return model, saved


def _score(predict, videos, **window):
    frame_mse, mask = per_video_horizon_mse(predict, videos, TEST_IDS, **window)
    return summarise(frame_mse, mask)


def _profile(name: str, path: Path, videos, device) -> dict:
    model, saved = _load(path, device)
    seconds = saved.get("epoch_seconds") or []
    predict = _device_predict(model, device, wrapped=False)
    original = _score(predict, videos)
    mirror = _score(predict, videos, mirror=True)
    rotation = _score(predict, videos, turns=1)
    speeds = {str(k): _score(predict, videos, k=k) for k in (1, 2, 3)}
    gap_m = float(consistency_gap(predict, videos, TEST_IDS, True, 0).mean())
    gap_r = float(consistency_gap(predict, videos, TEST_IDS, False, 1).mean())
    result = {
        "parameters": trainable_parameters(model),
        "hardware": saved.get("hardware"),
        "epoch_seconds_mean": float(sum(seconds) / len(seconds)) if seconds else None,
        "inference_seconds": time_model(model, videos, device, wrapped=False),
        "original": original,
        "EM": mirror,
        "ER": rotation,
        "CM": gap_m,
        "CR": gap_r,
        "speed": speeds,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _profile_wrapped(path: Path, videos, device) -> tuple[dict, torch.nn.Module]:
    model, saved = _load(path, device)
    seconds = saved.get("epoch_seconds") or []
    predict = _device_predict(model, device, wrapped=True)
    original = _score(predict, videos)
    mirror = _score(predict, videos, mirror=True)
    rotation = _score(predict, videos, turns=1)
    speeds = {str(k): _score(predict, videos, k=k) for k in (1, 2, 3)}
    gap_m = consistency_gap(predict, videos, TEST_IDS, True, 0)
    gap_r = consistency_gap(predict, videos, TEST_IDS, False, 1)
    summary = {
        "parameters": trainable_parameters(model),
        "hardware": saved.get("hardware"),
        "epoch_seconds_mean": float(sum(seconds) / len(seconds)) if seconds else None,
        "inference_seconds": time_model(model, videos, device, wrapped=True),
        "original": original,
        "EM": mirror,
        "ER": rotation,
        "CM": float(gap_m.mean()),
        "CR": float(gap_r.mean()),
        "speed": speeds,
    }
    return summary, model


def _figure6(tag: str, predict, videos, dest: Path, ids: dict, figure5_index: int) -> None:
    panels = []
    gif_dir = dest / "gifs"
    for kind, mirror, turns in (("mirror", True, 0), ("rotation", False, 1)):
        for rank, dataset_id in enumerate(ids[kind]):
            inputs, target = _one_video(videos, dataset_id, 1, 1)
            pred = predict(inputs.unsqueeze(0))[0]
            pred_t = predict(apply_group(inputs.unsqueeze(0), mirror, turns))[0]
            rows = forecast_rows(inputs, pred, target, mirror, turns, pred_t)
            title = f"{tag} {kind} video {dataset_id}"
            panels.append((title, rows))
            save_static_panel(rows, title, dest / f"figure6_{tag}_{kind}_{rank}.png")
            save_horizon_gif(rows, title, gif_dir / f"figure6_{tag}_{kind}_{rank}.gif")
    save_figure3(panels, dest / f"figure6a_{tag}.png")
    blocks = []
    for k, label in ((1, "1×"), (2, "2×"), (3, "3×")):
        inputs, target = _one_video(videos, figure5_index, 1, k)
        pred = predict(inputs.unsqueeze(0))[0]
        blocks.append((label, inputs, pred[:2], target[:2]))
    save_figure5(blocks, dest / f"figure6b_{tag}.png")


def _mean(rows: list[dict], *keys: str) -> float:
    total = 0.0
    for row in rows:
        item = row
        for key in keys:
            item = item[key]
        total += float(item)
    return total / len(rows)


def choose_final(scores: dict) -> str:
    """Pick symmetrised D, or symmetrised D+B when Table 5 supports it.

    D+B is used only when it does not give up more than 5% of D's original
    E(5) and it is better than D on both transformed errors. The comparison
    uses the learned models, which is what Table 5 reports.
    """
    e5_d = _mean(scores["D"], "original", "E5")
    e5_db = _mean(scores["DB"], "original", "E5")
    em_d = _mean(scores["D"], "EM", "E5")
    em_db = _mean(scores["DB"], "EM", "E5")
    er_d = _mean(scores["D"], "ER", "E5")
    er_db = _mean(scores["DB"], "ER", "E5")
    if e5_db <= 1.05 * e5_d and em_db < em_d and er_db < er_d:
        return "DB"
    return "D"


def main() -> None:
    root = output_root()
    dest = root / "outputs"
    dest.mkdir(parents=True, exist_ok=True)
    device = resolve_device()
    videos = load_videos()
    print(f"phase 2 output {root} device {device}", flush=True)
    for name in MODELS:
        for seed in (0, 1):
            train(name, seed, device=device, output_dir=root)

    scores = {}
    for name in ("A", *MODELS):
        per_seed = []
        for seed in (0, 1):
            path = _find(f"{name}_seed{seed}.pt")
            print(f"scoring {name} seed {seed} from {path}", flush=True)
            per_seed.append(_profile(name, path, videos, device))
        scores[name] = per_seed

    final_name = choose_final(scores)
    print(f"final underlying model: {final_name}", flush=True)
    ids = _read_json("figure3_ids.json")
    figure5_index = _read_json("video_ids.json")["figure5"]
    wrapped = {}
    for name in ("B", "D", "DB"):
        per_seed = []
        model = None
        for seed in (0, 1):
            path = _find(f"{name}_seed{seed}.pt")
            print(f"scoring symmetrised {name} seed {seed}", flush=True)
            summary, model = _profile_wrapped(path, videos, device)
            per_seed.append(summary)
            if seed == 0 and name == final_name:
                predict = _device_predict(model, device, wrapped=True)
                _figure6(name, predict, videos, dest, ids, figure5_index)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        wrapped[name] = per_seed

    payload = {"models": scores, "wrapped": wrapped, "final": final_name}
    (dest / "phase2_scores.json").write_text(json.dumps(payload, indent=2))
    print("phase 2 complete", flush=True)


if __name__ == "__main__":
    main()
