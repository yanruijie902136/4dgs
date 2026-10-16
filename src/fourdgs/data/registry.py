"""Explicit reader registration; datasets never depend on model or renderer code."""

from collections.abc import Callable

from fourdgs.config import DataConfig
from fourdgs.data.readers import colmap, dnerf, hypernerf, neu3d, panoptic
from fourdgs.data.types import SceneData

READERS: dict[str, Callable[..., SceneData]] = {
    "dnerf": dnerf.load,
    "neu3d": neu3d.load,
    "dynerf": neu3d.load,
    "hypernerf": hypernerf.load,
    "nerfies": hypernerf.load,
    "colmap": colmap.load,
    "panoptic": panoptic.load,
    "multiview": colmap.load_multiview,
}


def register_reader(name: str, reader: Callable[..., SceneData]) -> None:
    if name in READERS:
        raise ValueError(f"Reader already registered: {name}")
    READERS[name] = reader


def load_scene(config: DataConfig, *, load_points: bool = True) -> SceneData:
    """Read metadata and optionally initial points, without modifying the dataset."""
    if config.format not in READERS:
        raise ValueError(f"Unknown dataset format {config.format!r}; choose from {sorted(READERS)}")
    scene = READERS[config.format](config, load_points=load_points)
    if load_points and config.add_points:
        import numpy as np

        cloud = scene.point_cloud
        if cloud is None:
            raise ValueError("The reader did not provide an initial point cloud")
        extra = np.random.random((100000, 3)) * (
            cloud.points.max(0) - cloud.points.min(0)
        ) + cloud.points.min(0)
        cloud.points = np.vstack((cloud.points, extra.astype(np.float32)))
        cloud.colors = np.vstack((cloud.colors, np.random.random((100000, 3)).astype(np.float32)))
    return scene
