"""Versioned, atomic checkpoints with complete training continuation state."""

import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from fourdgs.models.gaussian import POINT_PARAMETERS, GaussianModel

FORMAT_VERSION = 1


def random_state() -> dict:
    np_state = np.random.get_state()
    if not isinstance(np_state, tuple):
        raise TypeError("Expected NumPy's legacy global random state")
    return {
        "python": random.getstate(),
        "numpy": (np_state[0], np_state[1].tolist(), *np_state[2:]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_random_state(state: dict) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
    )
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def save_checkpoint(
    path: str | Path,
    model: GaussianModel,
    config,
    *,
    stage: str,
    step: int,
    optimizer=None,
    density=None,
    sampler=None,
) -> None:
    """Save after a completed iteration, replacing the destination atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": FORMAT_VERSION,
        "config": OmegaConf.to_container(config, resolve=True),
        "model": model.state_dict(),
        "active_sh_degree": model.active_sh_degree,
        "spatial_lr_scale": model.spatial_lr_scale,
        "stage": stage,
        "step": step,
        "rng": random_state(),
        "optimizer": optimizer.optimizer.state_dict() if optimizer else None,
        "density": density.state_dict() if density else None,
        "sampler": sampler.state_dict() if sampler else None,
    }
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(handle)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> dict:
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or payload.get("version") != FORMAT_VERSION:
        raise ValueError(
            "Unsupported checkpoint format; use '4dgs import-legacy' for original snapshots"
        )
    return payload


def model_from_checkpoint(payload: dict, device: str | torch.device) -> GaussianModel:
    config = OmegaConf.create(payload["config"]["model"])
    model = GaussianModel(config).to(device)
    state = payload["model"]
    model.replace_points({name: state[name].to(device) for name in POINT_PARAMETERS})
    model.load_state_dict(state)
    model.active_sh_degree = payload["active_sh_degree"]
    model.spatial_lr_scale = payload["spatial_lr_scale"]
    return model
