import pytest
import torch

from fourdgs.io.checkpoint import model_from_checkpoint, read_checkpoint, save_checkpoint
from fourdgs.models.gaussian import GaussianModel
from fourdgs.training.losses import plane_smoothness


def test_deformation_disabled_is_identity(model):
    config = model.deformation.deformation_net.config
    for flag in ("no_dx", "no_ds", "no_dr", "no_do", "no_dshs"):
        setattr(config, flag, True)
    coarse, fine = model.attributes(0.4, "coarse"), model.attributes(0.4, "fine")
    for name in coarse.__dataclass_fields__:
        assert torch.equal(getattr(coarse, name), getattr(fine, name))


@pytest.mark.parametrize("grid_pe,no_grid", [(0, True), (1, False), (2, False)])
def test_optional_feature_modes(config, model, grid_pe, no_grid):
    config.model.grid_pe = grid_pe
    config.model.no_grid = no_grid
    other = GaussianModel(config.model)
    other.replace_points(
        {name: value.detach().clone() for name, value in model.point_parameters().items()}
    )
    values = other.attributes(0.3)
    assert values.positions.shape == (model.count, 3)
    assert torch.isfinite(values.positions).all()


def test_export_attributes_include_deformed_opacity(model):
    model.deformation.deformation_net.config.no_do = False
    head = model.deformation.deformation_net.opacity_deform[-1]
    with torch.no_grad():
        head.weight.zero_()
        head.bias.fill_(2)
    values = model.attributes(0.5)
    torch.testing.assert_close(values.opacity_logits, model.opacity_logits + 2)


def test_plane_second_difference():
    plane = torch.arange(4, dtype=torch.float32).square().reshape(1, 1, 4, 1)
    assert plane_smoothness(plane).item() == 4
    assert plane_smoothness(torch.ones(1, 1, 2, 3)).item() == 0


def test_occupancy_bounds_reload(config, model, tmp_path):
    config.model.empty_voxel = True
    original = GaussianModel(config.model)
    original.replace_points(
        {name: value.detach().clone() for name, value in model.point_parameters().items()}
    )
    original.deformation.deformation_net.set_aabb([2, 3, 4], [-2, -3, -4])
    path = tmp_path / "occupancy.pt"
    save_checkpoint(path, original, config, stage="fine", step=3)
    restored = model_from_checkpoint(read_checkpoint(path), "cpu")
    torch.testing.assert_close(
        restored.attributes(0.4).positions, original.attributes(0.4).positions
    )
