import importlib.util

import pytest
import torch

from fourdgs.data.types import Camera
from fourdgs.rendering.rasterizer import render

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or importlib.util.find_spec("diff_gaussian_rasterization") is None,
        reason="CUDA extensions/GPU unavailable",
    ),
]


@pytest.mark.parametrize("covariance", [False, True])
@pytest.mark.parametrize("colors", [False, True])
@pytest.mark.parametrize("stage", ["coarse", "fine"])
def test_rasterizer_forward_backward(model, config, covariance, colors, stage):
    model = model.cuda()
    with torch.no_grad():
        model.positions[:, 2] += 3
    config.render.compute_covariance_python = covariance
    config.render.convert_sh_python = colors
    camera = Camera(torch.eye(4), 32, 24, 30, 30, 16, 12, 0.2, "test")
    result = render(camera, model, config.render, torch.zeros(3, device="cuda"), stage)
    assert result.image.shape == (3, 24, 32)
    assert torch.isfinite(result.image).all()
    result.image.sum().backward()
    assert result.viewspace_points.grad is not None
    assert torch.isfinite(model.positions.grad).all()


def test_knn_matches_three_nearest_neighbors():
    from simple_knn._C import distCUDA2

    points = torch.rand(1500, 3, device="cuda")
    actual = distCUDA2(points)
    pairwise = torch.cdist(points, points).square()
    pairwise.fill_diagonal_(float("inf"))
    expected = pairwise.topk(3, largest=False).values.mean(1)
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-6)
