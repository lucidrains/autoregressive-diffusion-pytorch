import pytest
import torch
from torch import nn

from autoregressive_diffusion_pytorch.autoregressive_diffusion import ElucidatedDiffusion


class ZeroDenoiser(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.dim = dim

    def forward(self, noised_seq, *, times, cond):
        return torch.zeros_like(noised_seq)


def make_diffusion(num_sample_steps = 32):
    return ElucidatedDiffusion(
        dim = 4,
        net = ZeroDenoiser(4),
        num_sample_steps = num_sample_steps,
        sigma_min = 0.002,
        sigma_max = 80.,
        S_churn = 0.
    )


def test_single_step_schedule_is_finite_and_reaches_zero():
    diffusion = make_diffusion(num_sample_steps = 1)

    sigmas = diffusion.sample_schedule(1)

    assert sigmas.shape == (2,)
    assert torch.isfinite(sigmas).all()
    assert torch.allclose(sigmas, torch.tensor([80., 0.]))


def test_single_step_sampling_produces_finite_output():
    diffusion = make_diffusion(num_sample_steps = 1)

    sampled = diffusion.sample(cond = torch.zeros((2, 4)), num_sample_steps = 1)

    assert sampled.shape == (2, 4)
    assert torch.isfinite(sampled).all()


def test_invalid_sample_step_count_raises():
    diffusion = make_diffusion()

    with pytest.raises(ValueError, match = 'greater than zero'):
        diffusion.sample_schedule(0)


def test_multi_step_schedule_keeps_endpoints():
    diffusion = make_diffusion()

    sigmas = diffusion.sample_schedule(4)

    assert torch.isfinite(sigmas).all()
    assert torch.allclose(sigmas[[0, -2, -1]], torch.tensor([80., 0.002, 0.]))
    assert torch.all(sigmas[:-1].diff() < 0)
