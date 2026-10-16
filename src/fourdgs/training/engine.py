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

"""Coarse/fine training orchestration with explicit state and bounded data access."""

import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from fourdgs.config import validate
from fourdgs.data.registry import load_scene
from fourdgs.io.checkpoint import (
    model_from_checkpoint,
    read_checkpoint,
    restore_random_state,
    save_checkpoint,
)
from fourdgs.models.gaussian import GaussianModel
from fourdgs.rendering.rasterizer import render
from fourdgs.training.density import DensityController
from fourdgs.training.losses import training_loss
from fourdgs.training.optimizer import OptimizerController
from fourdgs.training.sampling import CameraSampler

LOGGER = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_step(samples, model, render_config, background, loss_config, stage):
    """Render a camera batch and accumulate its loss and screen-space gradients."""
    results = [render(sample.camera, model, render_config, background, stage) for sample in samples]
    if any(sample.image is None for sample in samples):
        raise ValueError("Training cameras must have target images")
    images = torch.stack([result.image for result in results])
    targets = torch.stack([sample.image.to(model.device) for sample in samples])
    loss, metrics = training_loss(images, targets, model, loss_config, stage)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Nonfinite loss during {stage} training")
    loss.backward()
    gradients = torch.zeros_like(results[0].viewspace_points)
    for result in results:
        gradient = result.viewspace_points.grad
        if gradient is None:
            raise RuntimeError("Rasterizer did not produce screen-space gradients")
        gradients = gradients + gradient
    visible = torch.stack([result.visible for result in results]).any(0)
    radii = torch.stack([result.radii for result in results]).max(0).values
    return metrics, gradients, visible, radii


@torch.no_grad()
def evaluate_samples(scene, model, config, background, stage) -> dict[str, float]:
    """Evaluate the original evenly spaced validation subset, safely handling empty splits."""
    result = {}
    for name in ("test", "train"):
        dataset = scene.split(name)
        if not len(dataset):
            continue
        values = []
        for index in range(10, 5000, 299):
            sample = dataset[index % len(dataset)]
            image = render(sample.camera, model, config.render, background, stage).image.clamp(0, 1)
            target = sample.image.to(model.device).clamp(0, 1)
            # Preserve the original training report's per-channel PSNR average.
            psnr = (-10 * torch.log10((image - target).square().flatten(1).mean(1))).mean()
            values.append(float(psnr))
        result[f"{name}_psnr"] = sum(values) / len(values)
    return result


def train(config) -> Path:
    """Train both stages, returning the final or explicitly interrupted checkpoint."""
    validate(config)
    if not str(config.run.device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Training requires an available CUDA device")
    seed_everything(config.run.seed)
    torch.set_num_threads(config.run.cpu_threads)
    output = Path(config.run.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = read_checkpoint(config.checkpoint.path) if config.checkpoint.path else None
    if config.checkpoint.resume and payload is None:
        raise ValueError("checkpoint.resume requires checkpoint.path")
    resume = payload if config.checkpoint.resume else None
    if resume:
        previous = resume["config"]
        if config.data.white_background != previous["data"]["white_background"]:
            raise ValueError("Exact resume requires unchanged data.white_background")
        for group in ("model", "loss", "optimization", "density"):
            if OmegaConf.to_container(config[group], resolve=True) != previous[group]:
                raise ValueError(f"Exact resume requires unchanged {group} configuration")
        for key in (
            "batch_size",
            "dataloader",
            "custom_sampler",
            "zerostamp_init",
            "iterations",
            "coarse_iterations",
        ):
            if config.training[key] != previous["training"][key]:
                raise ValueError(f"Exact resume requires unchanged training.{key}")
    scene = load_scene(config.data, load_points=payload is None)
    if payload:
        model = model_from_checkpoint(payload, config.run.device)
        if not config.checkpoint.resume:
            config.model.merge_with(payload["config"]["model"])
    else:
        if scene.point_cloud is None:
            raise ValueError("The reader did not provide an initial point cloud")
        model = GaussianModel(config.model)
        model.initialize(scene.point_cloud, scene.radius, config.run.device)
    OmegaConf.save(config, output / "config.yaml", resolve=True)
    background = torch.tensor([float(config.data.white_background)] * 3, device=model.device)
    writer = None
    if config.run.tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(output))
    from fourdgs.rendering.viewer import Viewer

    viewer = Viewer(config.viewer, config.data.root)
    log = (output / "training.jsonl").open("a")
    start = time.perf_counter()
    total_steps = 0
    torch.cuda.reset_peak_memory_stats(model.device)
    checkpoint = output / "checkpoints" / "latest.pt"
    sampling_state = None
    try:
        for stage, steps in (
            ("coarse", config.training.coarse_iterations),
            ("fine", config.training.iterations),
        ):
            if resume and resume["stage"] == "fine" and stage == "coarse":
                continue
            optimizer = OptimizerController(model, config.optimization)
            density = DensityController(model, optimizer, config.density)
            sampler = CameraSampler(
                scene.train,
                config.training,
                stage,
                config.run.seed,
                generator_state=torch.get_rng_state(),
            )
            if sampling_state is not None:
                sampler.rng.setstate(sampling_state["rng"])
                sampler.generator.set_state(sampling_state["generator"])
            first = 1
            if resume and stage == resume["stage"]:
                if any(resume.get(key) is None for key in ("optimizer", "density", "sampler")):
                    raise ValueError(
                        "This checkpoint contains weights only; use warm start instead of resume"
                    )
                optimizer.optimizer.load_state_dict(resume["optimizer"])
                density.load_state_dict(resume["density"])
                sampler.load_state_dict(resume["sampler"])
                first = resume["step"] + 1
                restore_random_state(resume["rng"])
            for step in range(first, steps + 1):
                viewer.service(scene, model, config.render, background, stage, step < steps)
                optimizer.update_learning_rates(step)
                if step % 1000 == 0:
                    model.increase_sh_degree()
                samples = sampler.next_batch()
                with torch.autograd.set_detect_anomaly(config.training.detect_anomaly):
                    metrics, gradients, visible, radii = train_step(
                        samples, model, config.render, background, config.loss, stage
                    )
                with torch.no_grad():
                    if step < config.density.densify_until_iter:
                        density.observe(gradients, visible, radii)
                        density.update(step, stage, scene.radius)
                    # Keep the upstream final-fine-step convention.
                    if step < config.training.iterations:
                        optimizer.step()
                    else:
                        optimizer.optimizer.zero_grad(set_to_none=True)
                total_steps += 1
                if step % config.training.log_interval == 0 or step == steps:
                    row = {
                        "stage": stage,
                        "step": step,
                        "points": model.count,
                        **{key: float(value) for key, value in metrics.items()},
                    }
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    LOGGER.info(
                        "%s %d/%d loss=%.6f PSNR=%.3f points=%d",
                        stage,
                        step,
                        steps,
                        row["loss"],
                        row["psnr"],
                        model.count,
                    )
                    if writer:
                        for key, value in metrics.items():
                            writer.add_scalar(f"{stage}/{key}", float(value), step)
                if step in config.training.test_iterations:
                    evaluation = evaluate_samples(scene, model, config, background, stage)
                    LOGGER.info("%s step %d: %s", stage, step, evaluation)
                    log.write(json.dumps({"stage": stage, "step": step, **evaluation}) + "\n")
                if config.training.render_process and (
                    (step < 1000 and step % 10 == 9)
                    or (step < 3000 and step % 50 == 49)
                    or (step < 60000 and step % 100 == 99)
                ):
                    from fourdgs.rendering.outputs import save_image

                    with torch.no_grad():
                        for name in ("train", "test"):
                            dataset = scene.split(name)
                            if len(dataset):
                                sample = dataset[step % len(dataset)]
                                save_image(
                                    output / "progress" / f"{stage}_{name}_{step:06d}.png",
                                    render(
                                        sample.camera, model, config.render, background, stage
                                    ).image,
                                )
                interrupted = (
                    config.training.stop_after and total_steps >= config.training.stop_after
                )
                save = step == steps or step in config.training.save_iterations or interrupted
                periodic = (
                    config.training.checkpoint_interval
                    and step % config.training.checkpoint_interval == 0
                )
                if save or periodic:
                    save_checkpoint(
                        checkpoint,
                        model,
                        config,
                        stage=stage,
                        step=step,
                        optimizer=optimizer,
                        density=density,
                        sampler=sampler,
                    )
                    if save and not interrupted:
                        save_checkpoint(
                            output / "checkpoints" / f"{stage}_{step:06d}.pt",
                            model,
                            config,
                            stage=stage,
                            step=step,
                            optimizer=optimizer,
                            density=density,
                            sampler=sampler,
                        )
                if interrupted:
                    return checkpoint
            sampling_state = sampler.state_dict()
    finally:
        torch.cuda.synchronize(model.device)
        resources = {
            "elapsed_seconds": time.perf_counter() - start,
            "completed_steps": total_steps,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(model.device),
        }
        (output / "resources.json").write_text(json.dumps(resources, indent=2))
        viewer.close()
        log.close()
        if writer:
            writer.close()
    return checkpoint
