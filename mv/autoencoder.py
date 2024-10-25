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


class EncoderLayer(torch.nn.Module):
    def __init__(self,
                 channels_in: int,
                 channels_out: int,
                 dropout: float,
                 padding: int = 0,
                 use_batch_norm: bool = True):
        super(EncoderLayer, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=channels_in, out_channels=channels_out, kernel_size=3, padding=padding)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        if use_batch_norm:
            self.batch_norm = torch.nn.BatchNorm2d(channels_out)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        if hasattr(self, 'batch_norm'):
            x = self.batch_norm(x)
        return x

class CrafterEnvEncoderV0(torch.nn.Module):

    def __init__(self,
                 channels_size: list[int],
                 latent_size: int,
                 dropout: float = 0.2,
                 use_batch_norm: bool = True):
        super(CrafterEnvEncoderV0, self).__init__()

        self.layer1 = EncoderLayer(len(objects),     channels_size[0], dropout, padding=1, use_batch_norm=use_batch_norm) #99 -> 99
        self.layer2 = EncoderLayer(channels_size[0], channels_size[1], dropout, padding=0, use_batch_norm=use_batch_norm) #99 -> 77
        self.layer3 = EncoderLayer(channels_size[1], channels_size[2], dropout, padding=1, use_batch_norm=use_batch_norm) #77 -> 77
        self.layer4 = EncoderLayer(channels_size[2], channels_size[3], dropout, padding=0, use_batch_norm=use_batch_norm) #77 -> 55
        self.layer5 = EncoderLayer(channels_size[3], channels_size[4], dropout, padding=0, use_batch_norm=use_batch_norm) #55 -> 33

        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(3*3*channels_size[4], latent_size)


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.flatten(x)
        x = self.linear(x)
        return x


class DecoderLayer(torch.nn.Module):

    def __init__(self,
                 channels_in,
                 channels_out,
                 padding=0,
                 activation=True):
        super(DecoderLayer, self).__init__()
        self.activation = activation
        self.deconv = torch.nn.ConvTranspose2d(
            in_channels=channels_in,
            out_channels=channels_out,
            padding=padding,
            kernel_size=3)

        if self.activation:
            self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.deconv(x)
        if self.activation:
            x = self.relu(x)
        return x

class CrafterEnvDecoderV0(torch.nn.Module):
    def __init__(self,
                 channels_size: list[int],
                 latent_size: int):
        super(CrafterEnvDecoderV0, self).__init__()
        self.linear = torch.nn.Linear(latent_size, 3*3*channels_size[4])
        self.unflatten = torch.nn.Unflatten(dim=1, unflattened_size=(channels_size[4], 3, 3))

        self.layer5 = DecoderLayer(channels_size[4], channels_size[3], padding=1)
        self.layer4 = DecoderLayer(channels_size[3], channels_size[2], padding=0)
        self.layer3 = DecoderLayer(channels_size[2], channels_size[1], padding=1)
        self.layer2 = DecoderLayer(channels_size[1], channels_size[0], padding=0)
        self.layer1 = DecoderLayer(channels_size[0], len(objects), padding=0, activation=False)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.unflatten(x)
        x = self.layer5(x)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = self.sigmoid(x)
        return x

class CrafterEnvAutoencoderV0(torch.nn.Module):
    def __init__(self,
                 channels_size: list[int],
                 latent_size: int,
                 dropout: float = 0.2,
                 use_batch_norm: bool = True):
        super(CrafterEnvAutoencoderV0, self).__init__()
        self.encoder = CrafterEnvEncoderV0(channels_size, latent_size, dropout, use_batch_norm)
        self.decoder = CrafterEnvDecoderV0(channels_size, latent_size)

    def forward(self, x):
        # x shape BS len(object_weights) 9 9
        x = self.encoder(x)
        norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x = torch.div(x, norm)
        x = self.decoder(x)
        return x


def create_autoencoder(config) -> CrafterEnvAutoencoderV0:
    hidden_channel_0 = int(config['hidden_channel_0'])
    hidden_channel_1 = int(config['hidden_channel_1'])
    hidden_channel_2 = int(config['hidden_channel_2'])
    hidden_channel_3 = int(config['hidden_channel_1'])
    hidden_channel_4 = int(config['hidden_channel_2'])
    latent_size = int(config['latent_size'])
    dropout = config['dropout']


    return CrafterEnvAutoencoderV0(
        channels_size=[hidden_channel_0, hidden_channel_1, hidden_channel_2, hidden_channel_3, hidden_channel_4],
        latent_size=latent_size,
        dropout=dropout)