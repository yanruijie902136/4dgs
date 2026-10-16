"""Streaming metrics with the original image normalization and LPIPS weights."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_msssim import ms_ssim

from fourdgs.evaluation.image import ssim


def load_image(path: Path, device) -> torch.Tensor:
    with Image.open(path) as image:
        return (
            torch.from_numpy(np.array(image.convert("RGB"), copy=True))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device)
            .float()
            / 255
        )


@torch.no_grad()
def evaluate(config) -> dict:
    """Read one image pair at a time and retain only scalar per-view results."""
    from fourdgs.evaluation.perceptual.lpips import LPIPS

    torch.set_num_threads(config.run.cpu_threads)
    models = (
        {name: LPIPS(name).to(config.run.device).eval() for name in ("vgg", "alex")}
        if config.perceptual_metrics
        else {}
    )
    root = Path(config.run.output_dir)
    summary, per_view = {}, {}
    for split in config.render.splits:
        directory = root / split
        render_dir, gt_dir = directory / "renders", directory / "gt"
        if not gt_dir.exists():
            continue
        paths = sorted(render_dir.glob("*.png"))
        if {p.name for p in paths} != {p.name for p in gt_dir.glob("*.png")}:
            raise ValueError(f"Rendered/ground-truth image names differ in {directory}")
        views = {}
        for path in paths:
            prediction = load_image(path, config.run.device)
            target = load_image(gt_dir / path.name, config.run.device)
            if prediction.shape != target.shape:
                raise ValueError(f"Image dimensions differ: {path.name}")
            mse = (prediction - target).square().mean()
            values = {
                "PSNR": float(-10 * torch.log10(mse)),
                "SSIM": float(ssim(prediction, target)),
            }
            if min(prediction.shape[-2:]) > 160:
                multiscale = float(ms_ssim(prediction, target, data_range=1, size_average=True))
                values.update({"MS-SSIM": multiscale, "D-SSIM": (1 - multiscale) / 2})
            for name, model in models.items():
                values[f"LPIPS-{name}"] = float(model(prediction, target))
            views[path.name] = values
        if views:
            summary[split] = {
                key: sum(view[key] for view in views.values()) / len(views)
                for key in next(iter(views.values()))
            }
            per_view[split] = views
    if not summary:
        raise ValueError(f"No paired rendered and ground-truth images under {root}")
    (root / "results.json").write_text(json.dumps(summary, indent=2))
    (root / "per_view.json").write_text(json.dumps(per_view, indent=2))
    return summary
