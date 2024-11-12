import torch
import math

from functorch.experimental import replace_all_batch_norm_modules_
import mv
import mv.utils
from mv.model.autoencoder import CrafterEnvEncoder2dV0, CrafterEnvDecoder2dV0

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

        print(f"Encoder1d channels: {channels}")
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
        print(f"Decoder1d channels: {channels}")
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




class Autoencoder2plus1(torch.nn.Module):

    def __init__(self,
        film_length: int,
        channels_size_2d,
        latent_size_2d,
        latent_size_3d,
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

if __name__ == "__main__":
    from itertools import chain

    model = Autoencoder2plus1(film_length=64)
    params = chain(model.encoder1d.parameters(), model.decoder1d.parameters())

    x = torch.randn(32, 64, len(mv.utils.objects), 9, 9)
    y = model(x)
    print(y.shape)
