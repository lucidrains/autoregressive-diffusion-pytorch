import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from x_transformers import (
    Decoder
)

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

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
