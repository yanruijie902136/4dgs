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

"""Optional SIBR network protocol adapter with explicit socket ownership."""

import json
import logging
import socket
from copy import copy
from dataclasses import replace

import torch

from fourdgs.config import ViewerConfig
from fourdgs.data.types import Camera
from fourdgs.rendering.rasterizer import render

LOGGER = logging.getLogger(__name__)


class ViewerCamera:
    """Adapt viewer matrices without discarding off-center projection terms."""

    def __init__(self, camera: Camera, view: torch.Tensor, projection: torch.Tensor):
        self.camera = camera
        self.view = view
        self.combined_projection = projection

    def __getattr__(self, name):
        return getattr(self.camera, name)

    def projection(self, device):
        return (
            self.view.to(device),
            self.combined_projection.to(device),
            self.view.inverse()[3, :3].to(device),
        )


def receive_exact(connection: socket.socket, count: int) -> bytes:
    """TCP may split either the header or payload across multiple receives."""
    chunks = bytearray()
    while len(chunks) < count:
        chunk = connection.recv(count - len(chunks))
        if not chunk:
            raise ConnectionError("Viewer disconnected")
        chunks.extend(chunk)
    return bytes(chunks)


class Viewer:
    """Serve the upstream viewer protocol only when explicitly enabled."""

    def __init__(self, config: ViewerConfig, source_path: str = ""):
        self.listener: socket.socket | None = None
        self.connection: socket.socket | None = None
        self.count = 0
        self.source_path = source_path
        if config.enabled:
            self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.listener.bind((config.host, config.port))
            self.listener.listen(1)
            self.listener.setblocking(False)

    def service(self, scene, model, config, background, stage, training: bool) -> None:
        if self.listener is None:
            return
        if self.connection is None:
            try:
                self.connection, _ = self.listener.accept()
                self.connection.settimeout(1.0)
            except BlockingIOError:
                return
        try:
            while self.connection:
                length = int.from_bytes(receive_exact(self.connection, 4), "little")
                if length > 1_000_000:
                    raise ValueError("Viewer message is too large")
                message = json.loads(receive_exact(self.connection, length))
                width, height = message["resolution_x"], message["resolution_y"]
                if not width or not height:
                    self._send_verification()
                    if message["train"] and (training or not message["keep_alive"]):
                        break
                    continue
                view = torch.tensor(message["view_matrix"], dtype=torch.float32).reshape(4, 4)
                view[:, 1:3] *= -1
                projection = torch.tensor(
                    message["view_projection_matrix"], dtype=torch.float32
                ).reshape(4, 4)
                projection[:, 1:3] *= -1
                camera = Camera(
                    view.T,
                    width,
                    height,
                    width / (2 * torch.tan(torch.tensor(message["fov_x"] / 2)).item()),
                    height / (2 * torch.tan(torch.tensor(message["fov_y"] / 2)).item()),
                    width / 2,
                    height / 2,
                    0,
                    "viewer",
                    message["z_near"],
                    message["z_far"],
                )
                if len(scene.video):
                    index = self.count % len(scene.video)
                    if (self.count // len(scene.video)) % 2:
                        index = len(scene.video) - index - 1
                    camera = replace(camera, time=scene.video.records[index].camera.time)
                self.count += 1

                options = copy(config)
                options.convert_sh_python = bool(message["shs_python"])
                options.compute_covariance_python = bool(message["rot_scale_python"])
                with torch.no_grad():
                    image = render(
                        ViewerCamera(camera, view, projection),
                        model,
                        options,
                        background,
                        stage,
                        scaling_modifier=message["scaling_modifier"],
                    ).image
                pixels = (
                    (image.clamp(0, 1) * 255)
                    .byte()
                    .permute(1, 2, 0)
                    .contiguous()
                    .cpu()
                    .numpy()
                    .tobytes()
                )
                self.connection.sendall(pixels)
                self._send_verification()
                if message["train"] and (training or not message["keep_alive"]):
                    break
        except (OSError, ValueError, KeyError, ConnectionError):
            LOGGER.exception("Viewer connection closed")
            if self.connection is not None:
                self.connection.close()
            self.connection = None

    def _send_verification(self) -> None:
        if self.connection is None:
            raise ConnectionError("Viewer disconnected")
        verification = self.source_path.encode("utf-8")
        self.connection.sendall(len(verification).to_bytes(4, "little") + verification)

    def close(self) -> None:
        for connection in (self.connection, self.listener):
            if connection is not None:
                connection.close()
