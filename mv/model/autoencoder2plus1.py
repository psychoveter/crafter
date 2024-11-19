import torch
import math

from functorch.experimental import replace_all_batch_norm_modules_
import mv
import mv.utils
from mv.model.autoencoder import CrafterEnvEncoder2dV0, CrafterEnvDecoder2dV0, load_model, load_tune_run


class Encoder1d(torch.nn.Module):

    def __init__(self,
                 length: int, # length of the frame sequence
                 in_channels: int,
                 latent_dim: int,
                 dropout: float = 0.3,
                 use_skip: bool = True
                 ):
        super().__init__()
        assert(math.log2(length).is_integer(), "Length must be power of 2")
        assert (in_channels < latent_dim, "In channels must be less than latent dim")
        n_layers = int(math.log2(length))
        channels_step = int((latent_dim - in_channels) / n_layers)
        channels = [in_channels + i * channels_step for i in range(n_layers)]
        channels.append(latent_dim)
        self.use_skip = use_skip

        print(f"Encoder1d channels: {channels}")
        layers = []
        for i in range(n_layers):
            layers.append(torch.nn.Conv1d(channels[i], channels[i+1], kernel_size=5, stride=2, padding=2))
            layers.append(torch.nn.Dropout(dropout))
            layers.append(torch.nn.BatchNorm1d(channels[i+1]))
            layers.append(torch.nn.ReLU())

        self.layers = torch.nn.Sequential(*layers)
        self.flatten = torch.nn.Flatten()
        self.ff = torch.nn.Linear(channels[-1], latent_dim)

        if use_skip:
            self.skip_ff = torch.nn.Linear(length * in_channels, channels[-1])

    def forward(self, x):
        # shape should be BS, C, L
        print(f"Encoder1 input shape: {x.shape}")

        if self.use_skip:
            x_skip = self.flatten(x)
            x_skip = self.skip_ff(x_skip)

        for i, layer in enumerate(self.layers):
            print(f"Encoder layer {i} shape {x.shape}")
            x = layer(x)
        # x = self.layers(x)

        x = self.flatten(x)

        if self.use_skip:
            x = x_skip + x

        x = self.ff(x)

        x = torch.nn.functional.relu(x)
        return x

class Decoder1d(torch.nn.Module):

    def __init__(self,
                 length: int,
                 latent_dim: int,
                 out_channels: int,
                 dropout: float = 0.05,
                 ):
        super().__init__()
        assert (latent_dim > out_channels)
        n_layers = int(math.log2(length))
        channels_step = int((latent_dim - out_channels) / n_layers)
        channels = [out_channels + i * channels_step for i in range(n_layers)]
        channels.append(latent_dim)

        # skip layer transforms (latent_dim) -> (out_channels * length)
        # then unflatten  (out_channels * length) -> (C, L)
        self.skip_ff = torch.nn.Linear(latent_dim, out_channels * length)
        self.skip_unflatten = torch.nn.Unflatten(dim=1, unflattened_size=(out_channels, length))

        self.unflatten = torch.nn.Unflatten(dim=1, unflattened_size=(latent_dim, 1))



        layers = []
        print(f"Decoder1d channels: {channels}")
        for i in reversed(range(n_layers + 1)):
            if not i == 0:
                # print(f"Channel {channels[i]} -> {channels[i - 1]}")
                # (BS, C_i, L) -> (BS, C_i, L)
                layers.append(torch.nn.ConvTranspose1d(channels[i], channels[i], kernel_size=5, padding=2, stride=1))
                layers.append(torch.nn.Dropout(dropout))
                layers.append(torch.nn.BatchNorm1d(channels[i]))
                layers.append(torch.nn.ReLU())
                # (BS, C_i, L) -> (BS, C_i+1, 2L)
                layers.append(torch.nn.ConvTranspose1d(channels[i], channels[i-1], kernel_size=5, padding=2, stride=2, output_padding=1))
                layers.append(torch.nn.Dropout(dropout))
                layers.append(torch.nn.BatchNorm1d(channels[i-1]))
                layers.append(torch.nn.ReLU())

        self.layers = torch.nn.Sequential(*layers)
        print(f"Channel {channels[0]} -> {out_channels}")
        self.out_conv = torch.nn.ConvTranspose1d(channels[0], out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x_skip = self.skip_ff(x)
        x_skip = self.skip_unflatten(x_skip)

        x = self.unflatten(x)
        for i, l in enumerate(self.layers):
            print(f"Decoder layer {i}, shape: {x.shape}")
            x = l(x)
        # x = self.layers(x)

        x = x_skip + x
        x = self.out_conv(x)
        x = torch.nn.functional.relu(x)

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

        self.decoder2d = CrafterEnvDecoder2dV0(
            channels_size=channels_size_2d,
            latent_size=latent_size_2d
        )

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

    def forward(self, x):
        assert(len(x.shape) == 5)
        b, f, c, h, w = x.shape

        assert(f == self.film_length)
        assert(c == len(mv.utils.objects))
        print(f"shape of input {x.shape}")
        x = x.view(-1, c, h, w)
        x = self.encoder2d(x)
        x = x.view(b, f, -1) # BS L C
        print(f"shape after encoder2d: {x.shape}")

        x = x.transpose(1, 2) # BS C L
        x = self.encoder1d(x)
        print(f"shape after encoder1d: {x.shape}")

        # norm = torch.norm(x, p=2, dim=1, keepdim=True)
        # x = torch.div(x, norm)

        x = self.decoder1d(x)
        print(f"shape after decoder1d: {x.shape}")

        x = x.contiguous()
        x = x.view(b * f, -1)
        x = self.decoder2d(x)
        print(f"shape after decoder2d: {x.shape}")

        x = x.view(b, f, c, h, w)
        return x

def load_ae_2plus1(ae2d_folder: str, film_length: int, latent_size_3d: int):
    ae_model_state, ae_params = load_tune_run(ae2d_folder)
    model = Autoencoder2plus1(
        film_length=film_length,
        channels_size_2d=[
            ae_params['hidden_channel_0'],
            ae_params['hidden_channel_1'],
            ae_params['hidden_channel_2'],
            ae_params['hidden_channel_3'],
            ae_params['hidden_channel_4'],
        ],
        latent_size_2d=ae_params['latent_size'],
        latent_size_3d=latent_size_3d,
    )

    def sub_state(name, state):
        return {k[len(name) + 1:]: state[k] for k in state.keys() if k.startswith(name)}

    encoder_state = sub_state('encoder', ae_model_state)
    decoder_state = sub_state('decoder', ae_model_state)
    model.encoder2d.load_state_dict(encoder_state)
    model.decoder2d.load_state_dict(decoder_state)

    return model

if __name__ == "__main__":
    film_length = 16
    model = load_ae_2plus1(
        '/Users/Oleg.Bukhvalov/projects/montevideo/crafter/mv/ray_results/autoencoder-0/TorchTrainer_c7c66_00000_0_2024-10-29_18-09-23',
        film_length=film_length,
        latent_size_3d=256,
    )

    # params = chain(model.encoder1d.parameters(), model.decoder1d.parameters())
    print("model has been loaded")

    x = torch.randn(4, film_length, len(mv.utils.objects), 9, 9)
    y = model(x)
    print(y.shape)
