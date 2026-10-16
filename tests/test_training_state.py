import random
from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from torch.utils.data import DataLoader

from fourdgs.data.registry import load_scene
from fourdgs.data.types import FrameDataset
from fourdgs.io.checkpoint import model_from_checkpoint, read_checkpoint, save_checkpoint
from fourdgs.io.ply import read_gaussians, write_gaussians
from fourdgs.training.density import DensityController
from fourdgs.training.optimizer import OptimizerController, exponential_lr
from fourdgs.training.sampling import CameraSampler


def populate_adam(model, optimizer):
    sum(parameter.square().sum() for parameter in model.point_parameters().values()).backward()
    optimizer.step()


def test_optimizer_topology_keeps_moments_and_groups(model, config):
    optimizer = OptimizerController(model, config.optimization)
    density = DensityController(model, optimizer, config.density)
    populate_adam(model, optimizer)
    moments = optimizer.optimizer.state[model.positions]["exp_avg"].clone()
    network_ids = [
        id(value)
        for group in optimizer.optimizer.param_groups
        if group["name"] in ("grid", "deformation")
        for value in group["params"]
    ]
    density.append(
        {name: value[:2].detach().clone() for name, value in model.point_parameters().items()}
    )
    assert model.count == 8
    assert torch.equal(optimizer.optimizer.state[model.positions]["exp_avg"][:6], moments)
    assert torch.count_nonzero(optimizer.optimizer.state[model.positions]["exp_avg"][6:]) == 0
    mask = torch.tensor([True, False, False, False, False, False, False, False])
    density.prune(mask)
    assert model.count == 7
    assert torch.equal(optimizer.optimizer.state[model.positions]["exp_avg"][:5], moments[1:])
    assert network_ids == [
        id(value)
        for group in optimizer.optimizer.param_groups
        if group["name"] in ("grid", "deformation")
        for value in group["params"]
    ]
    populate_adam(model, optimizer)


def test_opacity_reset_before_first_adam_step(model, config):
    optimizer = OptimizerController(model, config.optimization)
    optimizer.replace(
        {"opacity_logits": torch.full_like(model.opacity_logits, -4.0)}, reset_moments=True
    )
    populate_adam(model, optimizer)
    assert torch.isfinite(model.opacity_logits).all()


@pytest.mark.parametrize("dataloader", [False, True])
def test_sampler_resume_and_epoch_boundary(config, dnerf_root, dataloader):
    config.data.root = str(dnerf_root)
    config.training.batch_size = 3
    config.training.num_workers = 0
    config.training.dataloader = dataloader
    dataset = load_scene(config.data, load_points=False).train
    sampler = CameraSampler(dataset, config.training, "fine", 17)
    sampler.next_batch()
    state = deepcopy(sampler.state_dict())
    expected = [[sample.camera.name for sample in sampler.next_batch()] for _ in range(4)]
    restored = CameraSampler(dataset, config.training, "fine", 999)
    restored.load_state_dict(state)
    assert expected == [[sample.camera.name for sample in restored.next_batch()] for _ in range(4)]


def test_checkpoint_roundtrip_and_continuation(model, config, tmp_path):
    optimizer = OptimizerController(model, config.optimization)
    density = DensityController(model, optimizer, config.density)
    populate_adam(model, optimizer)
    path = tmp_path / "model.pt"
    save_checkpoint(
        path, model, config, stage="fine", step=12, optimizer=optimizer, density=density
    )
    payload = read_checkpoint(path)
    restored = model_from_checkpoint(payload, "cpu")
    for key, value in model.state_dict().items():
        assert torch.equal(restored.state_dict()[key], value), key
    resumed_optimizer = OptimizerController(restored, config.optimization)
    resumed_optimizer.optimizer.load_state_dict(payload["optimizer"])
    populate_adam(model, optimizer)
    populate_adam(restored, resumed_optimizer)
    assert torch.equal(model.positions, restored.positions)
    assert not list(tmp_path.glob(".*.pt.*"))


def test_gaussian_ply_roundtrip(model, tmp_path):
    path = tmp_path / "frame.ply"
    attributes = model.attributes(0, "coarse")
    write_gaussians(path, attributes)
    loaded = read_gaussians(path, model.config.sh_degree, "cpu")
    for name, value in model.point_parameters().items():
        assert torch.equal(value, loaded[name]), name


def test_schedule_endpoints():
    assert exponential_lr(0.1, 0.001, 0, 100) == pytest.approx(0.1)
    assert exponential_lr(0.1, 0.001, 100, 100) == pytest.approx(0.001)
    assert exponential_lr(0.1, 0.001, 50, 100) == pytest.approx(0.01)


def test_sampler_preserves_consumed_random_state(config, dnerf_root):
    config.data.root = str(dnerf_root)
    dataset = load_scene(config.data, load_points=False).train
    sampler = CameraSampler(dataset, config.training, "coarse", 17)
    expected = random.Random(17)
    remaining = list(range(len(dataset)))
    for _ in range(3):
        index = remaining.pop(expected.randint(0, len(remaining) - 1))
        assert sampler.next_batch()[0].camera.name == dataset[index].camera.name
        assert sampler.rng.getstate() == expected.getstate()


def test_sampler_matches_original_dataloader_rng(config, dnerf_root):
    config.data.root = str(dnerf_root)
    config.training.dataloader = True
    config.training.batch_size = 3
    config.training.num_workers = 0
    record = load_scene(config.data, load_points=False).train.records[0]
    dataset = FrameDataset(
        [replace(record, camera=replace(record.camera, name=str(i))) for i in range(11)], True
    )
    torch.manual_seed(17)
    torch.rand(137)  # Model initialization consumes the global CPU generator.
    initial_state = torch.get_rng_state()
    loader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=0, collate_fn=list)
    expected = [[frame.camera.name for frame in batch] for _ in range(3) for batch in loader]
    expected_state = torch.get_rng_state()

    sampler = CameraSampler(dataset, config.training, "coarse", 999, generator_state=initial_state)
    actual = [[frame.camera.name for frame in sampler.next_batch()] for _ in expected]
    assert actual == expected
    assert torch.equal(sampler.generator.get_state(), expected_state)
    assert torch.equal(torch.get_rng_state(), expected_state)  # Sampling stays isolated.
