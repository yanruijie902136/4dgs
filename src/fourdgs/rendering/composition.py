"""Compose any number of independently deformed scenes with similarity transforms."""

import math
from pathlib import Path

import imageio.v2 as imageio
import torch

from fourdgs.config import validate
from fourdgs.data.registry import load_scene
from fourdgs.geometry.gaussians import build_rotation, quaternion_multiply
from fourdgs.geometry.sh import evaluate_sh
from fourdgs.io.checkpoint import model_from_checkpoint, read_checkpoint
from fourdgs.models.gaussian import GaussianAttributes
from fourdgs.rendering.outputs import image_array, save_image
from fourdgs.rendering.rasterizer import rasterize


@torch.no_grad()
def compose_scenes(config) -> Path:
    """Render transformed models together through one depth-sorted splatting pass."""
    validate(config)
    torch.set_num_threads(config.run.cpu_threads)
    if not config.composition:
        raise ValueError("composition must contain at least one checkpoint and transform")
    scene = load_scene(config.data, load_points=False)
    models = []
    for item in config.composition:
        if item.scale <= 0 or len(item.rotation) != 4 or len(item.translation) != 3:
            raise ValueError(
                "Composition requires positive scale, wxyz rotation, and XYZ translation"
            )
        payload = read_checkpoint(item.checkpoint)
        model = model_from_checkpoint(payload, config.run.device)
        rotation = torch.tensor(item.rotation, device=model.device, dtype=torch.float32).reshape(
            1, 4
        )
        if rotation.norm() == 0:
            raise ValueError("Composition quaternion must be nonzero")
        models.append((item, model, payload["stage"], rotation, build_rotation(rotation)[0]))
    output = Path(config.run.output_dir) / "composition"
    output.mkdir(parents=True, exist_ok=True)
    background = torch.tensor([float(config.data.white_background)] * 3, device=config.run.device)
    for split in config.render.splits:
        records = scene.split(split).records
        if config.render.max_frames:
            records = records[: config.render.max_frames]
        if not records:
            continue
        directory = output / split
        directory.mkdir(parents=True, exist_ok=True)
        writer = (
            imageio.get_writer(directory / "video.mp4", fps=config.render.fps, macro_block_size=1)
            if config.render.video
            else None
        )
        try:
            for index, record in enumerate(records):
                attributes, colors = [], []
                for item, model, stage, quaternion, rotation in models:
                    values = model.attributes(record.camera.time + item.time_offset, stage)
                    positions = values.positions @ rotation.T * item.scale + torch.tensor(
                        item.translation, device=model.device
                    )
                    direction = torch.nn.functional.normalize(
                        (positions - record.camera.center.to(model.device)) @ rotation, dim=-1
                    )
                    colors.append(
                        (
                            evaluate_sh(
                                min(model.active_sh_degree, config.render.max_sh_degree),
                                values.sh.transpose(1, 2),
                                direction,
                            )
                            + 0.5
                        ).clamp_min(0)
                    )
                    attributes.append(
                        GaussianAttributes(
                            positions,
                            values.log_scales + math.log(item.scale),
                            quaternion_multiply(
                                quaternion.expand_as(values.rotations), values.rotations
                            ),
                            values.opacity_logits,
                            values.sh[:, :1],
                        )
                    )
                joined = GaussianAttributes(
                    *(
                        torch.cat([getattr(value, name) for value in attributes])
                        for name in GaussianAttributes.__dataclass_fields__
                    )
                )
                result = rasterize(
                    record.camera,
                    joined,
                    0,
                    background,
                    config.render,
                    override_color=torch.cat(colors),
                )
                save_image(directory / f"{index:05d}.png", result.image)
                if writer:
                    writer.append_data(image_array(result.image))
        finally:
            if writer:
                writer.close()
    return output
