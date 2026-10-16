"""Resumable camera sampling; prefetching never advances the committed cursor."""

import hashlib
import random
from collections import defaultdict
from collections.abc import Iterator

import torch
from torch.utils.data import DataLoader

from fourdgs.config import TrainingConfig
from fourdgs.data.types import FrameDataset, FrameSample


class CameraSampler:
    """Store the current epoch order and consumed cursor independently of workers."""

    def __init__(
        self,
        dataset: FrameDataset,
        config: TrainingConfig,
        stage: str,
        seed: int,
        *,
        generator_state: torch.Tensor | None = None,
    ):
        self.dataset, self.config = dataset, config
        self.rng = random.Random(seed)
        self.generator = torch.Generator().manual_seed(seed)
        if generator_state is not None:
            self.generator.set_state(generator_state.cpu())
        self.indices = list(range(len(dataset)))
        digest = hashlib.sha256()
        for record in dataset.records:
            camera = record.camera
            digest.update(
                repr(
                    (
                        camera.name,
                        camera.time,
                        camera.width,
                        camera.height,
                        camera.fx,
                        camera.fy,
                        camera.cx,
                        camera.cy,
                    )
                ).encode()
            )
            digest.update(camera.world_to_camera.numpy().tobytes())
        self.signature = digest.hexdigest()
        if config.zerostamp_init and stage == "coarse":
            self.indices = [
                i for i, record in enumerate(dataset.records) if record.camera.time == 0
            ]
        if not self.indices:
            raise ValueError("No cameras are available for the selected sampling policy")
        self.cursor = 0
        self.custom_pending = bool(config.custom_sampler and config.dataloader)
        self.order: list[int] = []
        self._loader: Iterator[list[FrameSample]] | None = None

    def _new_epoch(self) -> None:
        if self.custom_pending:
            by_time = defaultdict(list)
            for index in self.indices:
                by_time[self.dataset.records[index].camera.time].append(index)
            order: list[int] = []
            for time in sorted(by_time):
                values = by_time[time]
                # Preserve the original FineSampler's four draws and final selection.
                for _ in range(4):
                    permutation = torch.randperm(len(values), generator=self.generator).tolist()
                    current = []
                    for count, offset in enumerate(permutation, 1):
                        current.append(values[offset])
                        if count % 2 == 0 and len(order) > 2:
                            current.extend(self.rng.sample(order, 2))
                order.extend(current)
            self.order = order
            self.custom_pending = False
            self._draw_seed()  # DataLoader's worker base seed follows FineSampler construction.
        elif self.config.dataloader:
            # Upstream DataLoader draws a worker seed, then RandomSampler draws
            # a separate shuffle seed from the post-initialization CPU RNG.
            self._draw_seed()
            shuffle = torch.Generator().manual_seed(self._draw_seed())
            self.order = [
                self.indices[i]
                for i in torch.randperm(len(self.indices), generator=shuffle).tolist()
            ]
        else:
            self.order = self.indices.copy()
        self.cursor = 0
        self._loader = None

    def _draw_seed(self) -> int:
        return int(torch.empty((), dtype=torch.int64).random_(generator=self.generator))

    def next_batch(self) -> list[FrameSample]:
        if self.cursor >= len(self.order):
            self._new_epoch()
        if self.config.dataloader:
            if self._loader is None:
                remaining = self.order[self.cursor :]
                batches = [
                    remaining[i : i + self.config.batch_size]
                    for i in range(0, len(remaining), self.config.batch_size)
                ]
                loader = DataLoader(
                    self.dataset,
                    batch_sampler=batches,
                    num_workers=self.config.num_workers,
                    collate_fn=list,
                    generator=torch.Generator().manual_seed(0),
                )
                self._loader = iter(loader)
            result = next(self._loader)
            self.cursor += len(result)
            return result
        result = []
        while len(result) < self.config.batch_size:
            if self.cursor == len(self.order):
                self._new_epoch()
            # Draw only when consumed, preserving RNG state at stage boundaries.
            offset = self.rng.randint(self.cursor, len(self.order) - 1)
            self.order.insert(self.cursor, self.order.pop(offset))
            result.append(self.dataset[self.order[self.cursor]])
            self.cursor += 1
        return result

    def state_dict(self) -> dict:
        return {
            "order": self.order,
            "cursor": self.cursor,
            "rng": self.rng.getstate(),
            "generator": self.generator.get_state(),
            "custom_pending": self.custom_pending,
            "indices": self.indices,
            "signature": self.signature,
        }

    def load_state_dict(self, state: dict) -> None:
        if state["indices"] != self.indices or state["signature"] != self.signature:
            raise ValueError("Checkpoint sampler does not match this dataset")
        self.order, self.cursor = state["order"], state["cursor"]
        self.rng.setstate(state["rng"])
        self.generator.set_state(state["generator"].cpu())
        self.custom_pending = state["custom_pending"]
        self._loader = None
