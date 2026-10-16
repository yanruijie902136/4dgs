"""Export time-dependent Gaussians using the same deformation as rendering."""

import json
from pathlib import Path

import torch

from fourdgs.config import validate
from fourdgs.data.registry import load_scene
from fourdgs.io.checkpoint import model_from_checkpoint, read_checkpoint
from fourdgs.io.ply import write_gaussians


@torch.no_grad()
def export_frames(config) -> Path:
    """Write raw Gaussian snapshots; explicit times require no source dataset."""
    validate(config, needs_data=not bool(config.export.times))
    torch.set_num_threads(config.run.cpu_threads)
    if not config.checkpoint.path:
        raise ValueError("checkpoint.path is required")
    payload = read_checkpoint(config.checkpoint.path)
    model = model_from_checkpoint(payload, config.run.device).eval()
    times = list(config.export.times)
    if not times:
        scene = load_scene(config.data, load_points=False)
        times = [record.camera.time for record in scene.split(config.export.split).records]
    if not times:
        raise ValueError("No export timestamps; supply export.times or a nonempty split")
    output = Path(config.run.output_dir) / "gaussians"
    output.mkdir(parents=True, exist_ok=True)
    for index, time in enumerate(times):
        write_gaussians(output / f"time_{index:05d}.ply", model.attributes(time, payload["stage"]))
    (output / "timestamps.json").write_text(json.dumps(times, indent=2))
    return output
