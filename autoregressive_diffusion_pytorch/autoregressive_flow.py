from __future__ import annotations

import math
from math import sqrt
from typing import Literal
from functools import partial

import torch
from torch import nn, pi
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from torchdiffeq import odeint

import einx
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

from tqdm import tqdm

from x_transformers import Decoder

from autoregressive_diffusion_pytorch.autoregressive_diffusion import MLP

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

# gaussian diffusion

class Flow(Module):
    def __init__(
        self,
        dim: int,
        net: MLP,
        *,
        atol = 1e-5,
        rtol = 1e-5,
        method = 'midpoint'
    ):
        super().__init__()
        self.net = net
        self.dim = dim

        self.odeint_kwargs = dict(
            atol = atol,
            rtol = rtol,
            method = method
        )

    @property
    def device(self):
        return next(self.net.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        cond,
        num_sample_steps = 16
    ):

        batch = cond.shape[0]

        sampled_data_shape = (batch, self.dim)

        # start with random gaussian noise - y0

        noise = torch.randn(sampled_data_shape, device = self.device)

        # time steps

        times = torch.linspace(0., 1., num_sample_steps, device = self.device)

        # ode

        def ode_fn(t, x):
            t = repeat(t, '-> b', b = batch)
            flow = self.net(x, times = t, cond = cond)
            return flow

        trajectory = odeint(ode_fn, noise, times, **self.odeint_kwargs)

        sampled = trajectory[-1]

        return sampled

    # training

    def forward(self, seq, *, cond):
        batch_size, dim, device = *seq.shape, self.device

        assert dim == self.dim, f'dimension of sequence being passed in must be {self.dim} but received {dim}'

        times = torch.rand(batch_size, device = device)
        noise = torch.randn_like(seq)
        padded_times = right_pad_dims_to(seq, times)

        flow = seq - noise

        noised = (1.- padded_times) * noise + padded_times * seq

        pred_flow = self.net(noised, times = times, cond = cond)

        return F.mse_loss(pred_flow, flow)

# main model, a decoder with continuous wrapper + small denoising mlp

class AutoregressiveFlow(Module):
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
        flow_kwargs: dict = dict()
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

        self.flow = Flow(
            dim_input,
            self.denoiser,
            **flow_kwargs
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

            denoised_pred = self.flow.sample(cond = last_cond)

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

        return self.flow(target, cond = cond)

# image wrapper

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

class ImageAutoregressiveFlow(Module):
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

        self.model = AutoregressiveFlow(
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
