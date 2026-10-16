#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

"""Stream image/video exports with memory independent of sequence length."""

import json
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from fourdgs.config import validate
from fourdgs.data.registry import load_scene
from fourdgs.io.checkpoint import model_from_checkpoint, read_checkpoint
from fourdgs.rendering.rasterizer import render


def image_array(image: torch.Tensor) -> np.ndarray:
    """Round to PNG precision as torchvision.save_image did in the original code."""
    return (
        image.detach()
        .clamp(0, 1)
        .mul(255)
        .add(0.5)
        .clamp(0, 255)
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )


def save_image(path: str | Path, image: torch.Tensor) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_array(image)).save(path)


def load_render_context(config):
    validate(config)
    torch.set_num_threads(config.run.cpu_threads)
    if not config.checkpoint.path:
        raise ValueError("checkpoint.path is required")
    payload = read_checkpoint(config.checkpoint.path)
    model = model_from_checkpoint(payload, config.run.device).eval()
    config.model.merge_with(payload["config"]["model"])
    scene = load_scene(config.data, load_points=False)
    background = torch.tensor([float(config.data.white_background)] * 3, device=model.device)
    return model, scene, background, payload["stage"]


@torch.no_grad()
def render_dataset(config) -> Path:
    """Write each frame immediately; a video writer receives one CPU frame at a time."""
    model, scene, background, stage = load_render_context(config)
    output = Path(config.run.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output / "render_config.yaml", resolve=True)
    timings = {}
    for split in config.render.splits:
        frames = scene.split(split)
        count = (
            min(len(frames), config.render.max_frames) if config.render.max_frames else len(frames)
        )
        if not count:
            continue
        directory = output / split
        directory.mkdir(parents=True, exist_ok=True)
        writer = (
            imageio.get_writer(directory / "video.mp4", fps=config.render.fps, macro_block_size=1)
            if config.render.video
            else None
        )
        elapsed = 0.0
        try:
            for index in range(count):
                sample = frames[index]
                torch.cuda.synchronize(model.device)
                start = time.perf_counter()
                result = render(sample.camera, model, config.render, background, stage)
                torch.cuda.synchronize(model.device)
                elapsed += time.perf_counter() - start
                name = f"{index:05d}.png"
                save_image(directory / "renders" / name, result.image)
                if sample.image is not None:
                    save_image(directory / "gt" / name, sample.image)
                if writer:
                    # Original video encoding truncated instead of rounding.
                    writer.append_data(
                        (result.image.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
                    )
        finally:
            if writer:
                writer.close()
        timings[split] = {"frames": count, "render_seconds": elapsed, "fps": count / elapsed}
    (output / "render_times.json").write_text(json.dumps(timings, indent=2))
    return output
