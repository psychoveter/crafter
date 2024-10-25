from typing import Union, Callable, Tuple, Any, Optional, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import crafter
from mv.utils import create_nparr_onehot, objects, object_weights
import torch.nn.functional as F



class CrafterEnvDataset(Dataset):
    def __init__(self, env, size):
        self.env = env
        self.env.reset()

        self.items = []
        for i in range(size):
            self.env.step(self.env.action_space.sample())
            sample = create_nparr_onehot(self.env)
            sample = torch.tensor(sample, dtype=torch.float32)
            sample = sample.contiguous()
            self.items.append(sample)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

def create_datasets(train_size: int, test_size: int) -> Tuple[CrafterEnvDataset, CrafterEnvDataset]:
    env = crafter.Env()
    train_set = CrafterEnvDataset(env, size=train_size)
    test_set = CrafterEnvDataset(env, size=test_size)
    return train_set, test_set


class CrafterEnvAutoencoderV0(torch.nn.Module):
    def __init__(self,
                 channels_size: list[int],
                 latent_size: int,
                 dropout: float = 0.2):
        super(CrafterEnvAutoencoderV0, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=len(objects), out_channels=channels_size[0], kernel_size=3), # BS 7 7
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(channels_size[0]),
            torch.nn.Dropout(dropout),
            torch.nn.Conv2d(in_channels=channels_size[0], out_channels=channels_size[1], kernel_size=3), # BS 5 5
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(channels_size[1]),
            torch.nn.Dropout(dropout),
            torch.nn.Conv2d(in_channels=channels_size[1], out_channels=channels_size[2], kernel_size=3), # BS 3 3
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(channels_size[2]),
            torch.nn.Dropout(dropout),
            torch.nn.Flatten(),
            torch.nn.Linear(3*3*channels_size[2], latent_size),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_size, 3*3*channels_size[2]),
            torch.nn.Unflatten(dim=1, unflattened_size=(channels_size[2], 3, 3)),
            torch.nn.ConvTranspose2d(in_channels=channels_size[2], out_channels=channels_size[1], kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=channels_size[1], out_channels=channels_size[0], kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=channels_size[0], out_channels=len(objects), kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        # x shape BS len(object_weights) 9 9
        x = self.encoder(x)
        norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x = torch.div(x, norm)
        x = self.decoder(x)
        return x


def create_autoencoder(config) -> CrafterEnvAutoencoderV0:
    learning_rate = config['learning_rate']
    batch_size = int(config['batch_size'])
    hidden_channel_0 = int(config['hidden_channel_0'])
    hidden_channel_1 = int(config['hidden_channel_1'])
    hidden_channel_2 = int(config['hidden_channel_2'])
    latent_size = int(config['latent_size'])
    dropout = config['dropout']
    dataset_size = int(config['dataset_size'])
    max_epochs = int(config['max_epochs'])

    return CrafterEnvAutoencoderV0(
        channels_size=[hidden_channel_0, hidden_channel_1, hidden_channel_2],
        latent_size=latent_size,
        dropout=dropout)