import json
import os
from typing import Union, Callable, Tuple, Any, Optional, Dict

import pandas as pd
import torch
from torch.nn.functional import dropout
from torch.utils.data import DataLoader, Dataset
import crafter
from mv.utils import objects,sample_nparr_onehot



class CrafterEnvDataset(Dataset):
    def __init__(self, env, size, samples_from_world: int = 1000):
        self.env = env
        self.env.reset()

        self.items = []

        samples = sample_nparr_onehot(size, env, samples_from_world=samples_from_world)
        for sample in samples:
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
        self.dropout = torch.nn.Dropout(dropout)
        if use_batch_norm:
            self.batch_norm = torch.nn.BatchNorm2d(channels_out)
        self.relu = torch.nn.ReLU()

        # init weights
        # torch.nn.init.kaiming_normal_(self.conv.weight)


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
                 use_batch_norm: bool = True,
                 # hidden_skip_size: int = 16,
                 ):
        super(CrafterEnvEncoderV0, self).__init__()

        self.layer1 = EncoderLayer(len(objects),     channels_size[0], dropout, padding=1, use_batch_norm=use_batch_norm) #99 -> 99
        self.layer2 = EncoderLayer(channels_size[0], channels_size[1], dropout, padding=0, use_batch_norm=use_batch_norm) #99 -> 77
        self.layer3 = EncoderLayer(channels_size[1], channels_size[2], dropout, padding=1, use_batch_norm=use_batch_norm) #77 -> 77
        self.layer4 = EncoderLayer(channels_size[2], channels_size[3], dropout, padding=0, use_batch_norm=use_batch_norm) #77 -> 55
        self.layer5 = EncoderLayer(channels_size[3], channels_size[4], dropout, padding=0, use_batch_norm=use_batch_norm) #55 -> 33

        self.flatten = torch.nn.Flatten()


        self.linear_skip = torch.nn.Linear(len(objects) * 9 * 9, channels_size[4] * 3 * 3)
        self.bn_linear_skip = torch.nn.BatchNorm1d(channels_size[4] * 3 * 3)
        self.linear_out = torch.nn.Linear(channels_size[4] * 3 * 3, latent_size)
        self.bn_linear_out = torch.nn.BatchNorm1d(latent_size)

    def forward(self, x):
        source = x

        # convolution
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.flatten(x)

        #skip
        y = self.flatten(source)
        y = self.linear_skip(y)
        # y = self.bn_linear_skip(y)
        # y = torch.nn.functional.relu(y)
        x = x + y

        x = self.linear_out(x)
        x = torch.nn.functional.relu(x)
        # x = self.bn_linear_out(x)
        # x = torch.nn.functional.sigmoid(x)

        return x


class DecoderLayer(torch.nn.Module):

    def __init__(self,
                 channels_in,
                 channels_out,
                 padding=0,
                 dropout: float = 0.0,
                 activation=True):
        super(DecoderLayer, self).__init__()
        self.activation = activation

        self.deconv = torch.nn.ConvTranspose2d(
            in_channels=channels_in,
            out_channels=channels_out,
            padding=padding,
            kernel_size=3)

        self.conv = torch.nn.Conv2d(in_channels=channels_out, out_channels=channels_out, padding=1, kernel_size=3)

        self.dropout = torch.nn.Dropout(dropout)
        self.norm = torch.nn.BatchNorm2d(channels_out)


        #init weights
        # torch.nn.init.kaiming_normal_(self.deconv.weight)
        # torch.nn.init.kaiming_normal_(self.conv.weight)


    def forward(self, x):
        x = self.deconv(x)
        x = self.dropout(x)
        x = self.conv(x)
        x = self.dropout(x)
        x = self.norm(x)
        if self.activation:
            x = torch.nn.functional.relu(x)

        return x

class CrafterEnvDecoderV0(torch.nn.Module):
    def __init__(self,
                 channels_size: list[int],
                 latent_size: int,
                 use_skip: bool = True,
                 dropout: float = 0.2,
                 output_logit: bool = False
                 ):
        super(CrafterEnvDecoderV0, self).__init__()
        self.use_skip = use_skip
        self.output_logit = output_logit

        self.linear_conv = torch.nn.Linear(latent_size, channels_size[4] * 3 * 3)
        self.unflatten_conv = torch.nn.Unflatten(dim=1, unflattened_size=(channels_size[4], 3, 3))
        self.layer5 = DecoderLayer(channels_size[4], channels_size[3], padding=1) # c4 3 3 -> c3 3 3
        self.layer4 = DecoderLayer(channels_size[3], channels_size[2], padding=0) # c3 3 3 -> c2 5 5
        self.layer3 = DecoderLayer(channels_size[2], channels_size[1], padding=1) # c2 5 5 -> c1 5 5
        self.layer2 = DecoderLayer(channels_size[1], channels_size[0], padding=0) # c1 5 5 -> c0 7 7
        self.layer1 = DecoderLayer(channels_size[0], len(objects), padding=0) #c0 7 7 -> obj 9 9

        if use_skip:
            self.linear_skip = torch.nn.Linear(latent_size, len(objects) * 9 * 9)
            self.unflatten_skip = torch.nn.Unflatten(dim=1, unflattened_size=(len(objects), 9, 9))

        self.layer_out = DecoderLayer(len(objects), len(objects), padding=1, activation=False) #obj 9 9 -> obj 9 9
        if not self.output_logit:
            self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        source = x
        # deconvolution branch
        x = self.linear_conv(x)
        x = self.unflatten_conv(x)
        x = self.layer5(x)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)

        # skip branch
        if self.use_skip:
            y = self.linear_skip(source)
            y = torch.nn.functional.relu(y)
            y = self.unflatten_skip(y)
            x = x + y

        # out layer
        x = self.layer_out(x)

        if not self.output_logit:
            x = self.sigmoid(x)

        return x

class CrafterEnvAutoencoderV0(torch.nn.Module):
    def __init__(self,
                 channels_size: list[int],
                 latent_size: int,
                 encoder_dropout: float = 0.2,
                 decoder_dropout: float = 0.2,
                 use_batch_norm: bool = True,
                 output_logit: bool = False):
        super(CrafterEnvAutoencoderV0, self).__init__()
        self.encoder = CrafterEnvEncoderV0(channels_size,
                                           latent_size=latent_size,
                                           dropout=encoder_dropout,
                                           use_batch_norm=use_batch_norm)
        self.decoder = CrafterEnvDecoderV0(channels_size,
                                           latent_size=latent_size,
                                           dropout=decoder_dropout,
                                           output_logit=output_logit)

    def forward(self, x):
        # x shape BS len(object_weights) 9 9
        x = self.encoder(x)
        norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x = torch.div(x, norm)
        x = self.decoder(x)
        return x


def create_autoencoder(config, output_logits: bool = False) -> CrafterEnvAutoencoderV0:
    hidden_channel_0 = int(config['hidden_channel_0'])
    hidden_channel_1 = int(config['hidden_channel_1'])
    hidden_channel_2 = int(config['hidden_channel_2'])
    hidden_channel_3 = int(config['hidden_channel_3'])
    hidden_channel_4 = int(config['hidden_channel_4'])
    latent_size = int(config['latent_size'])
    encoder_dropout = config['encoder_dropout']
    decoder_dropout = config['decoder_dropout']


    return CrafterEnvAutoencoderV0(
        channels_size=[hidden_channel_0, hidden_channel_1, hidden_channel_2, hidden_channel_3, hidden_channel_4],
        latent_size=latent_size,
        encoder_dropout=encoder_dropout,
        decoder_dropout=decoder_dropout,
        output_logit=output_logits
    )


def load_tune_run(run_folder, checkpoint: str = None):
    """

    :param run_folder: absolute path to the folder with a single train inside ray tune
    :param checkpoint: is not None load specific checkpoint, otherwise load the  best checkpoint
    :return:
    """

    progress = pd.read_csv(os.path.join(run_folder, "progress.csv"))
    # load best loss checkpoint
    if checkpoint is not None:
        checkpoint_dir_name = os.path.join(run_folder, checkpoint, 'model.pt')
    else:
        sorted = progress.sort_values(by='loss', ascending=True)
        checkpoint_dir_name: str = sorted.values[0][2]
        checkpoint_dir_name: str = os.path.join(run_folder, checkpoint_dir_name, 'model.pt')

    print(checkpoint_dir_name)
    model_state = torch.load(checkpoint_dir_name)
    params_file = os.path.join(run_folder, 'params.json')
    with open(params_file) as f:
        params = json.load(f)
        if 'train_loop_config' in params:
            params = params['train_loop_config']
        print(params)
    return model_state, params

def load_model(run_folder, checkpoint=None):
    model_state, params = load_tune_run(run_folder, checkpoint)
    model = create_autoencoder(params)
    model.load_state_dict(model_state)
    return model
