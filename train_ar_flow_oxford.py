# hf datasets for easy oxford flowers training

import torchvision.transforms as T
from torch.utils.data import Dataset
from datasets import load_dataset

class OxfordFlowersDataset(Dataset):
    def __init__(
        self,
        image_size
    ):
        self.ds = load_dataset('nelorth/oxford-flowers')['train']

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.PILToTensor()
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        pil = self.ds[idx]['image']
        tensor = self.transform(pil)
        return tensor / 255.

flowers_dataset = OxfordFlowersDataset(
    image_size = 64
)

from autoregressive_diffusion_pytorch import ImageAutoregressiveFlow, ImageTrainer

model = ImageAutoregressiveFlow(
    model = dict(
        dim = 1024,
        depth = 8,
        heads = 8,
        mlp_depth = 4,
        decoder_kwargs = dict(
            rotary_pos_emb = True
        )
    ),
    image_size = 64,
    patch_size = 8,
    model_output_clean = True
)

trainer = ImageTrainer(
    model,
    dataset = flowers_dataset,
    num_train_steps = 1_000_000,
    learning_rate = 7e-5,
    batch_size = 32
)

trainer()
