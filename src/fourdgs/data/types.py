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

"""CPU-side scene contracts shared by every dataset reader."""

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class RenderCamera(Protocol):
    """Structural camera interface, also implemented by the viewer adapter."""

    @property
    def width(self) -> int: ...

    @property
    def height(self) -> int: ...

    @property
    def fx(self) -> float: ...

    @property
    def fy(self) -> float: ...

    @property
    def time(self) -> float: ...

    def projection(self, device: torch.device | str) -> tuple[Tensor, Tensor, Tensor]: ...


@dataclass(frozen=True)
class Camera:
    """Pinhole camera with a column-vector world-to-camera matrix, looking +Z.

    Rasterizer matrices are transposed only at the rendering boundary. Pixel
    intrinsics correspond to ``width`` and ``height``, after image resizing.
    """

    world_to_camera: Tensor
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    time: float
    name: str
    near: float = 0.01
    far: float = 100.0

    @property
    def center(self) -> Tensor:
        return self.world_to_camera.inverse()[:3, 3]

    def projection(self, device: torch.device | str) -> tuple[Tensor, Tensor, Tensor]:
        """Return transposed view, combined projection, and camera center."""
        view = self.world_to_camera.to(device).T.contiguous()
        projection = torch.zeros((4, 4), dtype=view.dtype, device=device)
        projection[0, 0] = 2 * self.fx / self.width
        projection[1, 1] = 2 * self.fy / self.height
        projection[0, 2] = (2 * self.cx - self.width) / self.width
        projection[1, 2] = (2 * self.cy - self.height) / self.height
        projection[2, 2] = self.far / (self.far - self.near)
        projection[2, 3] = -self.far * self.near / (self.far - self.near)
        projection[3, 2] = 1
        return view, view @ projection.T, view.inverse()[3, :3]


@dataclass(frozen=True)
class FrameRecord:
    """Lightweight metadata; no image pixels or CUDA allocations."""

    camera: Camera
    image_path: Path | None = None
    mask_path: Path | None = None
    composite_alpha: bool = False
    resize_filter: str = "lanczos"


@dataclass
class FrameSample:
    camera: Camera
    image: Tensor | None
    mask: Tensor | None = None


class FrameDataset(Dataset):
    """Read frames on demand with a bounded, per-process CPU image cache."""

    def __init__(self, records: list[FrameRecord], white_background: bool, cache_size: int = 32):
        self.records = records
        self.white_background = white_background
        self.cache_size = max(cache_size, 0)
        self._cache: OrderedDict[int, FrameSample] = OrderedDict()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> FrameSample:
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        record = self.records[index]
        camera = record.camera
        image_tensor = None
        if record.image_path is not None:
            with Image.open(record.image_path) as source:
                image = source.convert("RGBA" if record.composite_alpha else "RGB")
                if record.composite_alpha:
                    rgba = np.asarray(image, dtype=np.float64) / 255.0
                    rgb = rgba[..., :3] * rgba[..., 3:] + float(self.white_background) * (
                        1 - rgba[..., 3:]
                    )
                    # Preserve the original compositing quantization before resizing.
                    image = Image.fromarray((rgb * 255).astype(np.uint8))
                size = (camera.width, camera.height)
                if image.size != size:
                    method = (
                        Image.Resampling.BICUBIC
                        if record.resize_filter == "bicubic"
                        else Image.Resampling.LANCZOS
                    )
                    image = image.resize(size, method)
                image_tensor = (
                    torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255
                )
        mask = None
        if record.mask_path is not None:
            with Image.open(record.mask_path) as source:
                array = np.array(
                    source.convert("L").resize((camera.width, camera.height)), copy=True
                )
                mask = torch.from_numpy(array).unsqueeze(0).float() / 255
        sample = FrameSample(camera, image_tensor, mask)
        if self.cache_size:
            self._cache[index] = sample
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return sample


@dataclass
class PointCloud:
    """World-space XYZ and RGB in [0, 1], each shaped [N, 3]."""

    points: np.ndarray
    colors: np.ndarray


@dataclass
class SceneData:
    train: FrameDataset
    test: FrameDataset
    video: FrameDataset
    point_cloud: PointCloud | None
    radius: float
    format: str

    def split(self, name: str) -> FrameDataset:
        if name not in ("train", "test", "video"):
            raise ValueError(f"Unknown split: {name}")
        return getattr(self, name)
