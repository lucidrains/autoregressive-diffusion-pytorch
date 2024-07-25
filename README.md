<img src="./ar-diffusion.png" width="400px"></img>

## Autoregressive Diffusion - Pytorch (wip)

Implementation of the architecture behind <a href="https://arxiv.org/abs/2406.11838">Autoregressive Image Generation without Vector Quantization</a> in Pytorch

You can discuss the paper temporarily [here](https://discord.com/invite/9myQVTbN)

## Install

```bash
$ pip install autoregressive-diffusion-pytorch
```

## Usage

```python
import torch
from autoregressive_diffusion_pytorch import AutoregressiveDiffusion

model = AutoregressiveDiffusion(
    dim = 512,
    max_seq_len = 32
)

seq = torch.randn(3, 32, 512)

loss = model(seq)
loss.backward()

sampled = model.sample(batch_size = 3)

assert sampled.shape == seq.shape

```

For images treated as a sequence of tokens (as in paper)

```python
import torch
from autoregressive_diffusion_pytorch import (
    ImageAutoregressiveDiffusion
)

model = ImageAutoregressiveDiffusion(
    model = dict(
        dim = 512
    ),
    image_size = 64,
    patch_size = 8
)

images = torch.randn(3, 3, 64, 64)

loss = model(images)
loss.backward()

sampled = model.sample(batch_size = 3)

assert sampled.shape == images.shape

```

## Citations

```bibtex
@article{Li2024AutoregressiveIG,
    title   = {Autoregressive Image Generation without Vector Quantization},
    author  = {Tianhong Li and Yonglong Tian and He Li and Mingyang Deng and Kaiming He},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2406.11838},
    url     = {https://api.semanticscholar.org/CorpusID:270560593}
}
```
