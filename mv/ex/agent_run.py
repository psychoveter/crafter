#%%
import math

import crafter
from stable_baselines3 import PPO

import mv.utils
from crafter.engine import LocalView
from mv.autoencoder import CrafterEnvEncoder2dV0, CrafterEnvDecoder2dV0
from mv.datagen import CrafterAgentDataset3D

root = '/Users/Oleg.Bukhvalov/projects/montevideo/crafter'

model = PPO.load(f"{root}/ppo.zip")

env = crafter.Env()
# env = crafter.Recorder(env=env, directory=f"{root}/mv/agent_results")
env.reset()




#%%
import torch
from functorch.experimental import replace_all_batch_norm_modules_


x = torch.randn(32, 100, 29, 9, 9)
enc = CrafterEnvEncoder2dV0(channels_size=[16,16,16,16,16], latent_size=32)
replace_all_batch_norm_modules_(enc)

film_enc = torch.vmap(
    func=enc,
    randomness='different'
)
output = film_enc(x)

print(output.shape)

#%%
import torch
import math
from functorch.experimental import replace_all_batch_norm_modules_

class Encoder1d(torch.nn.Module):

    def __init__(self,
                 length: int,
                 in_channels: int,
                 latent_dim: int
                 ):
        super().__init__()
        assert(math.log2(length).is_integer(), "Length must be power of 2")
        assert (in_channels < latent_dim, "In channels must be less than latent dim")
        n_layers = int(math.log2(length))
        channels_step = int((latent_dim - in_channels) / n_layers)
        channels = [in_channels + i * channels_step for i in range(n_layers)]
        channels.append(latent_dim)

        print(f"{channels}")
        layers = []
        for i in range(n_layers):
            layers.append(torch.nn.Conv1d(channels[i], channels[i+1], kernel_size=5, stride=2, padding=2))
            layers.append(torch.nn.BatchNorm1d(channels[i+1]))
            layers.append(torch.nn.ReLU())

        self.layers = torch.nn.Sequential(*layers)
        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        x = self.layers(x)
        x = self.flatten(x)
        return x

e1d = Encoder1d(128, 16, 128)
print(e1d)
x = torch.randn(32, 16, 128)
y = e1d(x)
print(y.shape)


#%%
class Decoder1d(torch.nn.Module):
    def __init__(self, length: int, latent_dim: int, out_channels: int):
        super().__init__()
        assert (latent_dim > out_channels)
        n_layers = int(math.log2(length))
        channels_step = int((latent_dim - out_channels) / n_layers)
        channels = [out_channels + i * channels_step for i in range(n_layers)]
        channels.append(latent_dim)

        self.unflatten = torch.nn.Unflatten(dim=1, unflattened_size=(latent_dim, 1))

        layers = []
        print(channels)
        for i in reversed(range(n_layers + 1)):
            if i == 0:
                print(f"Channel {channels[0]} -> {out_channels}")
                layers.append(torch.nn.ConvTranspose1d(channels[0], out_channels, kernel_size=3, padding=1))
            else:
                print(f"Channel {channels[i]} -> {channels[i - 1]}")
                layers.append(torch.nn.ConvTranspose1d(channels[i], channels[i-1], kernel_size=5, padding=2, stride=2, output_padding=1))
                layers.append(torch.nn.BatchNorm1d(channels[i-1]))
                layers.append(torch.nn.ReLU())

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.unflatten(x)
        # for l in self.layers:
        #     print(x.shape)
        #     x = l(x)
        x = self.layers(x)
        return x

d1d = Decoder1d(128, 128, 16)
# print(d1d)
x_hat = d1d(y)
print(x_hat.shape)

#%%
import mv
import mv.utils
from mv.autoencoder import CrafterEnvEncoder2dV0, CrafterEnvDecoder2dV0

class Autoencoder2plus1(torch.nn.Module):

    def __init__(self,
        film_length: int,
        channels_size_2d = [16, 16, 16, 16, 16],
        latent_size_2d = 32,
        latent_size_3d = 128,
    ):
        super(Autoencoder2plus1, self).__init__()

        self.film_length = film_length

        self.encoder2d = CrafterEnvEncoder2dV0(
            channels_size=channels_size_2d,
            latent_size=latent_size_2d
        )
        replace_all_batch_norm_modules_(self.encoder2d) # support vmap


        self.decoder2d = CrafterEnvDecoder2dV0(
            channels_size=channels_size_2d,
            latent_size=latent_size_2d
        )
        replace_all_batch_norm_modules_(self.decoder2d) # support vmap

        self.encoder1d = Encoder1d(
            length=film_length,
            in_channels=latent_size_2d,
            latent_dim=latent_size_3d
        )

        self.decoder1d = Decoder1d(
            length=film_length,
            latent_dim=latent_size_3d,
            out_channels=latent_size_2d
        )
        self.enc_2d_vmap = torch.vmap(func=self.encoder2d, randomness='different')
        self.dec_2d_vmap = torch.vmap(func=self.decoder2d, in_dims=2, randomness='different')



    def forward(self, x):
        assert(len(x.shape) == 5)
        b, f, c, h, w = x.shape
        assert(f == self.film_length)
        assert(c == len(mv.utils.objects))


        x = self.enc_2d_vmap(x)
        x = x.transpose(1, 2)
        x = self.encoder1d(x)
        print(f"shape after encoder1d: {x.shape}")
        x = self.decoder1d(x)
        print(f"shape after decoder1d: {x.shape}")
        x = self.dec_2d_vmap(x)
        x = x.transpose(0, 1)

        return x


# cad = CrafterAgentDataset3D(env, model, 1, 100)
# sample = cad.__getitem__(0)
length = 128
enc21 = Autoencoder2plus1(
    film_length=length,
    latent_size_2d=48,
    latent_size_3d=128
)
x = torch.randn(32, length, 29, 9, 9)
y = enc21(x)
print(y.shape)