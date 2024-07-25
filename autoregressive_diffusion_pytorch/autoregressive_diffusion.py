from typing import Literal
from functools import partial

import torch
from torch import nn, pi
from torch.special import expm1
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

from x_transformers import (
    ContinuousTransformerWrapper,
    Decoder
)

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# tensor helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim

    if padding_dims <= 0:
        return t

    return t.view(*t.shape, *((1,) * padding_dims))

def pack_one(t, pattern):
    packed, ps = pack([t], pattern)

    def unpack_one(to_unpack, unpack_pattern = None):
        unpacked, = unpack(to_unpack, ps, default(unpack_pattern, pattern))
        return unpacked

    return packed, unpack_one

# sinusoidal embedding

class AdaptiveLayerNorm(Module):
    def __init__(
        self,
        dim,
        dim_condition = None
    ):
        super().__init__()
        dim_condition = default(dim_condition, dim)

        self.ln = nn.LayerNorm(dim, elementwise_affine = False)
        self.to_gamma = nn.Linear(dim_condition, dim, bias = False)
        nn.init.zeros_(self.to_gamma.weight)

    def forward(self, x, *, condition):
        normed = self.ln(x)
        gamma = self.to_gamma(condition)
        return normed * (gamma + 1.)

class LearnedSinusoidalPosEmb(Module):
    def __init__(self, dim):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# simple mlp

class MLP(Module):
    def __init__(
        self,
        dim,
        dim_cond,
        depth = 3,
        width = 1024,
        dropout = 0.
    ):
        super().__init__()
        layers = ModuleList([])

        self.to_time_emb = nn.Sequential(
            LearnedSinusoidalPosEmb(dim_cond),
            nn.Linear(dim_cond + 1, dim_cond),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_cond, dim_cond)
        )

        for _ in range(depth):

            adaptive_layernorm = AdaptiveLayerNorm(
                dim,
                dim_condition = dim_cond
            )

            block = nn.Sequential(
                nn.Linear(dim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim)
            )

            layers.append(ModuleList([
                adaptive_layernorm,
                block
            ]))

        self.layers = layers

    def forward(
        self,
        noised,
        *,
        times,
        cond
    ):

        time_emb = self.to_time_emb(times)
        cond = time_emb + cond

        denoised = noised

        for adaln, block in self.layers:
            residual = denoised
            denoised = adaln(denoised, condition = cond)
            denoised = block(denoised) + residual

        return denoised

# gaussian diffusion

def simple_linear_schedule(t, clip_min = 1e-9):
    return (1. - t).clamp(min = clip_min)

def cosine_schedule(t, start = 0, end = 1, tau = 1, clip_min = 1e-9):
    start, end, tau = map(torch.tensor, (start, end, tau))
    power = 2 * tau
    v_start = torch.cos(start * pi / 2) ** power
    v_end = torch.cos(end * pi / 2) ** power
    output = torch.cos((t * (end - start) + start) * pi / 2) ** power
    output = (v_end - output) / (v_end - v_start)
    return output.clamp(min = clip_min)

def gamma_to_alpha_sigma(gamma, scale = 1):
    return torch.sqrt(gamma) * scale, torch.sqrt(1 - gamma)

def gamma_to_log_snr(gamma, scale = 1, eps = 1e-5):
    return log(gamma * (scale ** 2) / (1. - gamma), eps = eps)

class GaussianDiffusion(Module):
    def __init__(
        self,
        model: MLP,
        *,
        timesteps = 1000,
        use_ddim = True,
        noise_schedule: Literal['linear', 'cosine'] = 'cosine',
        objective: Literal['eps', 'v'] = 'v',
        schedule_kwargs: dict = dict(),
        min_snr_loss_weight = True,
        min_snr_gamma = 5,
    ):
        super().__init__()
        self.model = model
        self.objective = objective

        if noise_schedule == 'linear':
            self.gamma_schedule = simple_linear_schedule
        elif noise_schedule == 'cosine':
            self.gamma_schedule = cosine_schedule
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')

        # gamma schedules

        self.gamma_schedule = partial(self.gamma_schedule, **schedule_kwargs)

        self.timesteps = timesteps
        self.use_ddim = use_ddim

        # min snr loss weight

        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma

    @property
    def device(self):
        return next(self.model.parameters()).device

    def get_sampling_timesteps(self, batch, *, device):
        times = torch.linspace(1., 0., self.timesteps + 1, device = device)
        times = repeat(times, 't -> b t', b = batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim = 0)
        times = times.unbind(dim = -1)
        return times

    @torch.no_grad()
    def ddpm_sample(self, shape):
        batch, device = shape[0], self.device

        time_pairs = self.get_sampling_timesteps(batch, device = device)

        img = torch.randn(shape, device=device)

        x_start = None
        last_latents = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step', total = self.timesteps):

            noise_cond = time

            # get predicted x0

            model_output, last_latents = self.model(maybe_normalized_img, noise_cond)

            # get log(snr)

            gamma = self.gamma_schedule(time)
            gamma_next = self.gamma_schedule(time_next)
            gamma, gamma_next = map(partial(right_pad_dims_to, img), (gamma, gamma_next))

            # get alpha sigma of time and next time

            alpha, sigma = gamma_to_alpha_sigma(gamma)
            alpha_next, sigma_next = gamma_to_alpha_sigma(gamma_next)

            # calculate x0 and noise

            if self.objective == 'eps':
                x_start = safe_div(img - sigma * model_output, alpha)

            elif self.objective == 'v':
                x_start = alpha * img - sigma * model_output

            # clip x0

            x_start.clamp_(-1., 1.)

            # derive posterior mean and variance

            log_snr, log_snr_next = map(gamma_to_log_snr, (gamma, gamma_next))

            c = -expm1(log_snr - log_snr_next)

            mean = alpha_next * (img * (1 - c) / alpha + c * x_start)
            variance = (sigma_next ** 2) * c
            log_variance = log(variance)

            # get noise

            noise = torch.where(
                rearrange(time_next > 0, 'b -> b 1 1 1'),
                torch.randn_like(img),
                torch.zeros_like(img)
            )

            img = mean + (0.5 * log_variance).exp() * noise

        return img

    @torch.no_grad()
    def ddim_sample(self, shape):
        batch, device = shape[0], self.device

        time_pairs = self.get_sampling_timesteps(batch, device = device)

        img = torch.randn(shape, device = device)

        x_start = None
        last_latents = None

        for times, times_next in tqdm(time_pairs, desc = 'sampling loop time step'):

            # get times and noise levels

            gamma = self.gamma_schedule(times)
            gamma_next = self.gamma_schedule(times_next)

            padded_gamma, padded_gamma_next = map(partial(right_pad_dims_to, img), (gamma, gamma_next))

            alpha, sigma = gamma_to_alpha_sigma(padded_gamma)
            alpha_next, sigma_next = gamma_to_alpha_sigma(padded_gamma_next)

            # predict x0

            maybe_normalized_img = self.maybe_normalize_img_variance(img)
            model_output, last_latents = self.model(maybe_normalized_img, times, x_start, last_latents, return_latents = True)

            # calculate x0 and noise

            if self.objective == 'eps':
                x_start = safe_div(img - sigma * model_output, alpha)

            elif self.objective == 'v':
                x_start = alpha * img - sigma * model_output

            # clip x0

            x_start.clamp_(-1., 1.)

            # get predicted noise

            pred_noise = safe_div(img - alpha * x_start, sigma)

            # calculate x next

            img = x_start * alpha_next + pred_noise * sigma_next

        return unnormalize_img(img)

    @torch.no_grad()
    def sample(self, *, cond, batch_size = 16):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.ddpm_sample if not self.use_ddim else self.ddim_sample
        return sample_fn((batch_size, channels, image_size, image_size))

    def forward(self, seq, *args, cond, **kwargs):

        batch, device = seq.shape[0], seq.device

        # sample random times

        times = torch.rand((batch,), device = device)

        # noise sample

        noise = torch.randn_like(seq)

        gamma = self.gamma_schedule(times)
        padded_gamma = right_pad_dims_to(seq, gamma)
        alpha, sigma =  gamma_to_alpha_sigma(padded_gamma)

        noised_seq = alpha * seq + sigma * noise

        # predict and take gradient step

        pred = self.model(noised_seq, times = times, cond = cond)

        if self.objective == 'eps':
            target = noise

        elif self.objective == 'v':
            target = alpha * noise - sigma * seq

        loss = F.mse_loss(pred, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        # min snr loss weight

        snr = (alpha * alpha) / (sigma * sigma)
        maybe_clipped_snr = snr.clone()

        if self.min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = self.min_snr_gamma)

        if self.objective == 'eps':
            loss_weight = maybe_clipped_snr / snr

        elif self.objective == 'v':
            loss_weight = maybe_clipped_snr / (snr + 1)

        return (loss * loss_weight).mean()

# main model, a decoder with continuous wrapper + small denoising mlp

class AutoregressiveDiffusion(Module):
    def __init__(
        self,
        dim,
        *,
        max_seq_len,
        depth = 8,
        dim_head = 64,
        heads = 8,
        mlp_depth = 3,
        mlp_width = 1024,
        decoder_kwargs: dict = dict(),
        mlp_kwargs: dict = dict(),
        diffusion_kwargs: dict = dict()
    ):
        super().__init__()

        self.start_token = nn.Parameter(torch.zeros(dim))
        self.max_seq_len = max_seq_len
        self.abs_pos_emb = nn.Embedding(max_seq_len, dim)

        self.transformer = Decoder(
            dim = dim,
            depth = depth,
            heads = heads,
            attn_dim_head = dim_head,
            **decoder_kwargs
        )

        self.denoiser = MLP(
            dim = dim,
            dim_cond = dim,
            depth = mlp_depth,
            width = mlp_width,
            **mlp_kwargs
        )

        self.diffusion = GaussianDiffusion(
            self.denoiser,
            **diffusion_kwargs
        )

    def forward(
        self,
        seq
    ):
        b = seq.shape[0]

        # append start tokens

        start_token = repeat(self.start_token, 'd -> b 1 d', b = b)
        seq = torch.cat((start_token, seq), dim = 1)

        # break into seq and the continuous targets to be predicted

        seq, target = seq[:, :-1], seq[:, 1:]

        cond = self.transformer(seq)

        # pack batch and sequence dimensions, so to train each token with different noise levels

        target, _ = pack_one(target, '* d')
        cond, _ = pack_one(cond, '* d')

        diffusion_loss = self.diffusion(target, cond = cond)

        return diffusion_loss
