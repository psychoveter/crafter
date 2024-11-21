from stable_baselines3 import PPO

from mv.datagen import CrafterAgentDataset3D
from mv.draw_utils import render_film_np_onehot, render_tensor_films
from mv.model.autoencoder import load_tune_run
from mv.model.autoencoder2plus1 import Autoencoder2plus1
import torch

# path = '/Users/Oleg.Bukhvalov/projects/montevideo/crafter/mv/ray_results/ae-2plus1-0/TorchTrainer_8f527_00000_0_2024-11-19_13-39-31'
path = '/Users/Oleg.Bukhvalov/projects/montevideo/crafter/mv/ray_results_remote/ae-2plus1-0/TorchTrainer_87048_00000_0_2024-11-19_14-29-23'


model_state, params = load_tune_run(path)

ae2d_folder = params['ae2d_folder']
_, ae2d_params = load_tune_run(ae2d_folder)

model = Autoencoder2plus1(
        film_length=params['film_length'],
        channels_size_2d=[
            ae2d_params['hidden_channel_0'],
            ae2d_params['hidden_channel_1'],
            ae2d_params['hidden_channel_2'],
            ae2d_params['hidden_channel_3'],
            ae2d_params['hidden_channel_4'],
        ],
        latent_size_2d=ae2d_params['latent_size'],
        latent_size_3d=params['latent_size_3d'],
)
model.load_state_dict(model_state)
model.eval()

print("Try model shape")
x = torch.randn(4, params['film_length'], 29, 9, 9)
y = model(x)

import crafter

env = crafter.Env()
root = '/Users/Oleg.Bukhvalov/projects/montevideo/crafter'
ppo_agent = PPO.load(f"{root}/ppo.zip")
dataset = CrafterAgentDataset3D(env = env, model = ppo_agent, dataset_size=1, film_length=params['film_length'])


def sample_and_show_film():
    film = dataset[0]
    batch = torch.Tensor(film)
    batch = batch.unsqueeze(0)

    print(f"Film batch shape: {batch.shape}")
    restored = model(batch)
    print(f"Restored shape: {restored.shape}")
    render_tensor_films(batch[0], restored[0], env)



film = dataset[0]

x = model.encoder2d(film)
# norm = torch.norm(x, p=2, dim=1, keepdim=True)
# x = torch.div(x, norm)
x = model.decoder2d(x)

print(x.shape)

render_tensor_films(film, x, env)