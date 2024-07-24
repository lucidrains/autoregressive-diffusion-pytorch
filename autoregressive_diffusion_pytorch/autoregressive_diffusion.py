import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

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
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
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
            nn.SiLU()
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

        for adaln, bloc in self.layers:
            residual = denoised
            denoised = adaln(denoised, condition = cond)
            denoised = block(denoised) + denoised

        return denoised

# main model, a decoder with continuous wrapper + small denoising mlp

class ContinuousDecoderWithMLPDenoiser(Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        max_seq_len,
        dim = 512,
        depth = 8,
        dim_head = 64,
        heads = 8,
        mlp_depth = 3,
        mlp_width = 1024,
        decoder_kwargs: dict = dict(),
        mlp_kwargs: dict = dict()
    ):
        super().__init__()

        self.transformer = ContinuousTransformerWrapper(
            dim_in = dim_in,
            dim_out = dim_out,
            max_seq_len = max_seq_len,
            attn_layers = Decoder(
                dim = dim,
                depth = depth,
                heads = heads,
                attn_dim_head = dim_head,
                **decoder_kwargs
            )
        )

        self.denoiser = MLP(
            dim = dim_in,
            dim_cond = dim_in,
            depth = mlp_depth,
            width = mlp_width,
            **mlp_kwargs
        )

    def forward(
        self,
        x,
        *,
        times,
    ):
        x = self.transformer(x)
        return x

# main class

class AutoregressiveDiffusion(Module):
    def __init__(
        self,        
        model: Module,
    ):
        super().__init__()
        self.model = model

    def forward(self, data):
        return data
