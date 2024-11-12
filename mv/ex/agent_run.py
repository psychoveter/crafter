#%%
import crafter
from stable_baselines3 import PPO
from mv.model.autoencoder import CrafterEnvEncoder2dV0
from mv.model.autoencoder2plus1 import Decoder1d, Encoder1d, Autoencoder2plus1

root = '/Users/Oleg.Bukhvalov/projects/montevideo/crafter'

#%%
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

e1d = Encoder1d(128, 16, 128)
print(e1d)
x = torch.randn(32, 16, 128)
y = e1d(x)
print(y.shape)

d1d = Decoder1d(128, 128, 16)
# print(d1d)
x_hat = d1d(y)
print(x_hat.shape)

#%%

# cad = CrafterAgentDataset3D(env, model, 1, 100)
# sample = cad.__getitem__(0)
length = 256
enc21 = Autoencoder2plus1(
    film_length=length,
    latent_size_2d=48,
    latent_size_3d=128
)
x = torch.randn(32, length, 29, 9, 9)
y = enc21(x)
print(y.shape)