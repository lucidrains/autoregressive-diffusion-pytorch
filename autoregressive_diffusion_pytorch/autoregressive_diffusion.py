from __future__ import annotations

import math
from typing import Literal
from functools import partial

import torch
from torch import nn, pi
from torch.special import expm1
import torch.nn.functional as F
from torch.nn import Module, ModuleList

import einx
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

from tqdm import tqdm

from x_transformers import Decoder

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

def safe_div(num, den, eps = 1e-5):
    return num / den.clamp(min = eps)

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
        dim_cond,
        dim_input,
        depth = 3,
        width = 1024,
        dropout = 0.
    ):
        super().__init__()
        layers = ModuleList([])

        self.to_time_emb = nn.Sequential(
            LearnedSinusoidalPosEmb(dim_cond),
            nn.Linear(dim_cond + 1, dim_cond),
        )

        for _ in range(depth):

            adaptive_layernorm = AdaptiveLayerNorm(
                dim_input,
                dim_condition = dim_cond
            )

            block = nn.Sequential(
                nn.Linear(dim_input, width),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(width, dim_input)
            )

            block_out_gamma = nn.Linear(dim_cond, dim_input, bias = False)
            nn.init.zeros_(block_out_gamma.weight)

            layers.append(ModuleList([
                adaptive_layernorm,
                block,
                block_out_gamma
            ]))

        self.layers = layers

    def forward(
        self,
        noised,
        *,
        times,
        cond
    ):
        assert noised.ndim == 2

        time_emb = self.to_time_emb(times)
        cond = F.silu(time_emb + cond)

        denoised = noised

        for adaln, block, block_out_gamma in self.layers:
            residual = denoised
            denoised = adaln(denoised, condition = cond)

            block_out = block(denoised) * (block_out_gamma(cond) + 1.)
            denoised = block_out + residual

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
        dim: int,
        model: MLP,
        *,
        timesteps = 1000,
        sampling_timesteps: int | None = None,
        use_ddim = False,
        noise_schedule: Literal['linear', 'cosine'] = 'cosine',
        objective: Literal['eps', 'v'] = 'v',
        schedule_kwargs: dict = dict(),
        clip_during_sampling = True,
        min_snr_loss_weight = True,
        min_snr_gamma = 5,
    ):
        super().__init__()
        self.model = model
        self.dim = dim
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

        # sampling related

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        self.use_ddim = use_ddim
        self.clip_during_sampling = clip_during_sampling

        # min snr loss weight

        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma

    @property
    def device(self):
        return next(self.model.parameters()).device

    def get_sampling_timesteps(self, batch, *, device):
        times = torch.linspace(1., 0., self.sampling_timesteps + 1, device = device)
        times = repeat(times, 't -> t b', b = batch)
        times = torch.stack((times[:-1], times[1:]), dim = 1)
        return times

    @torch.no_grad()
    def ddpm_sample(self, cond):
        batch, device = cond.shape[0], self.device

        time_pairs = self.get_sampling_timesteps(batch, device = device)

        seq = torch.randn((batch, self.dim), device = device)

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step', total = self.timesteps, leave = False):

            # get predicted x0

            model_output = self.model(seq, times = time, cond = cond)

            # get log(snr)

            gamma = self.gamma_schedule(time)
            gamma_next = self.gamma_schedule(time_next)

            gamma, gamma_next = map(partial(right_pad_dims_to, seq), (gamma, gamma_next))

            # get alpha sigma of time and next time

            alpha, sigma = gamma_to_alpha_sigma(gamma)
            alpha_next, sigma_next = gamma_to_alpha_sigma(gamma_next)

            # calculate x0 and noise

            if self.objective == 'eps':
                x_start = safe_div(seq - sigma * model_output, alpha)

            elif self.objective == 'v':
                x_start = alpha * seq - sigma * model_output

            # clip x0

            if self.clip_during_sampling:
                x_start.clamp_(-1., 1.)

            # derive posterior mean and variance

            log_snr, log_snr_next = map(gamma_to_log_snr, (gamma, gamma_next))

            c = -expm1(log_snr - log_snr_next)

            mean = alpha_next * (seq * (1 - c) / alpha + c * x_start)
            variance = (sigma_next ** 2) * c
            log_variance = log(variance)

            # get noise

            noise = einx.where(
                'b, b d, -> b d',
                time_next > 0,
                torch.randn_like(seq),
                0.
            )

            seq = mean + (0.5 * log_variance).exp() * noise

        return seq

    @torch.no_grad()
    def ddim_sample(self, cond):
        batch, device = cond.shape[0], self.device

        time_pairs = self.get_sampling_timesteps(batch, device = device)

        seq = torch.randn((batch, self.dim), device = device)

        for times, times_next in tqdm(time_pairs, desc = 'sampling loop time step', leave = False):

            # get times and noise levels

            gamma = self.gamma_schedule(times)
            gamma_next = self.gamma_schedule(times_next)

            padded_gamma, padded_gamma_next = map(partial(right_pad_dims_to, seq), (gamma, gamma_next))

            alpha, sigma = gamma_to_alpha_sigma(padded_gamma)
            alpha_next, sigma_next = gamma_to_alpha_sigma(padded_gamma_next)

            # predict x0

            model_output = self.model(seq, times = times, cond = cond)

            # calculate x0 and noise

            if self.objective == 'eps':
                x_start = safe_div(seq - sigma * model_output, alpha)

            elif self.objective == 'v':
                x_start = alpha * seq - sigma * model_output

            # clip x0

            if self.clip_during_sampling:
                x_start.clamp_(-1., 1.)

            # get predicted noise

            pred_noise = safe_div(seq - alpha * x_start, sigma)

            # calculate x next

            seq = x_start * alpha_next + pred_noise * sigma_next

        return seq

    @torch.no_grad()
    def sample(self, *, cond, batch_size = 16):
        sample_fn = self.ddpm_sample if not self.use_ddim else self.ddim_sample
        return sample_fn(cond)

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
        mlp_width = None,
        dim_input = None,
        decoder_kwargs: dict = dict(),
        mlp_kwargs: dict = dict(),
        diffusion_kwargs: dict = dict(
            timesteps = 1000,
            sampling_timesteps = 100,
            use_ddim = False,
            clip_during_sampling = True
        )
    ):
        super().__init__()

        self.start_token = nn.Parameter(torch.zeros(dim))
        self.max_seq_len = max_seq_len
        self.abs_pos_emb = nn.Embedding(max_seq_len, dim)

        dim_input = default(dim_input, dim)
        self.dim_input = dim_input
        self.proj_in = nn.Linear(dim_input, dim)

        self.transformer = Decoder(
            dim = dim,
            depth = depth,
            heads = heads,
            attn_dim_head = dim_head,
            **decoder_kwargs
        )

        self.denoiser = MLP(
            dim_cond = dim,
            dim_input = dim_input,
            depth = mlp_depth,
            width = default(mlp_width, dim),
            **mlp_kwargs
        )

        self.diffusion = GaussianDiffusion(
            dim_input,
            self.denoiser,
            **diffusion_kwargs
        )

    @property
    def device(self):
        return next(self.transformer.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        batch_size = 1,
        prompt  = None
    ):
        self.eval()

        start_tokens = repeat(self.start_token, 'd -> b 1 d', b = batch_size)

        if not exists(prompt):
            out = torch.empty((batch_size, 0, self.dim_input), device = self.device, dtype = torch.float32)
        else:
            out = prompt

        cache = None

        for _ in tqdm(range(self.max_seq_len - out.shape[1]), desc = 'tokens'):

            cond = self.proj_in(out)

            cond = torch.cat((start_tokens, cond), dim = 1)
            cond = cond + self.abs_pos_emb(torch.arange(cond.shape[1], device = self.device))

            cond, cache = self.transformer(cond, cache = cache, return_hiddens = True)

            last_cond = cond[:, -1]

            denoised_pred = self.diffusion.sample(cond = last_cond)

            denoised_pred = rearrange(denoised_pred, 'b d -> b 1 d')
            out = torch.cat((out, denoised_pred), dim = 1)

        return out

    def forward(
        self,
        seq
    ):
        b, seq_len, dim = seq.shape

        assert dim == self.dim_input
        assert seq_len == self.max_seq_len

        # break into seq and the continuous targets to be predicted

        seq, target = seq[:, :-1], seq

        # append start tokens

        seq = self.proj_in(seq)
        start_token = repeat(self.start_token, 'd -> b 1 d', b = b)

        seq = torch.cat((start_token, seq), dim = 1)
        seq = seq + self.abs_pos_emb(torch.arange(seq_len, device = self.device))

        cond = self.transformer(seq)

        # pack batch and sequence dimensions, so to train each token with different noise levels

        target, _ = pack_one(target, '* d')
        cond, _ = pack_one(cond, '* d')

        diffusion_loss = self.diffusion(target, cond = cond)

        return diffusion_loss

# image wrapper

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

class ImageAutoregressiveDiffusion(Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        channels = 3,
        model: dict = dict(),
    ):
        super().__init__()
        assert divisible_by(image_size, patch_size)

        num_patches = (image_size // patch_size) ** 2
        dim_in = channels * patch_size ** 2

        self.image_size = image_size
        self.patch_size = patch_size

        self.to_tokens = Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1 = patch_size, p2 = patch_size)

        self.model = AutoregressiveDiffusion(
            **model,
            dim_input = dim_in,
            max_seq_len = num_patches
        )

        self.to_image = Rearrange('b (h w) (c p1 p2) -> b c (h p1) (w p2)', p1 = patch_size, p2 = patch_size, h = int(math.sqrt(num_patches)))

    def sample(self, batch_size = 1):
        tokens = self.model.sample(batch_size = batch_size)
        images = self.to_image(tokens)
        return unnormalize_to_zero_to_one(images)

    def forward(self, images):
        images = normalize_to_neg_one_to_one(images)
        tokens = self.to_tokens(images)
        return self.model(tokens)
