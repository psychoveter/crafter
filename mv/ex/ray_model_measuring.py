#%%
import os.path

import pandas as pd
import json
import ray
import torch
import matplotlib
import PIL.Image as Image
import crafter
from mv.autoencoder import create_autoencoder, CrafterEnvAutoencoderV0
from mv.utils import create_nparr_onehot, render_nparr_onehot, draw_image_grid

ray_model_path = '/Users/Oleg.Bukhvalov/projects/montevideo/crafter/mv/ray_results/train_autoencoder_2024-10-24_13-09-05/train_autoencoder_00808_00028_28_batch_size=16,dataset_size=500,dropout=0.4159,hidden_channel_0=32,hidden_channel_1=128,hidden_cha_2024-10-24_13-09-06'

progress = pd.read_csv(os.path.join(ray_model_path, "progress.csv"))

# load checkpoint
sorted = progress.sort_values(by='loss', ascending=True)
checkpoint_dir_name: str = sorted.values[0][2]
checkpoint_dir_name: str = os.path.join(ray_model_path, checkpoint_dir_name, 'model.pt')
print(checkpoint_dir_name)
model_state = torch.load(checkpoint_dir_name)


params_file = os.path.join(ray_model_path, 'params.json')
with open(params_file) as f:
    params = json.load(f)
    print(params)

# params['dropout'] = 0
model = create_autoencoder(params)
model.load_state_dict(model_state)


env = crafter.Env()
env.reset()
env.step(env.action_space.sample())
np_sample = create_nparr_onehot(env)
print(np_sample.shape)

torch_sample = torch.tensor(np_sample, dtype=torch.float32).unsqueeze(dim=0)

print(torch_sample.shape)
restored = model(torch_sample).detach().numpy().round()
print(f"restored: {restored.shape}")

img_source = render_nparr_onehot(np_sample, env)
img_restored = render_nparr_onehot(restored[0], env)
img = draw_image_grid([img_source, img_restored])


Image.fromarray(img).show()





def accuracy(model: CrafterEnvAutoencoderV0, num_samples: int):
    env = crafter.Env()
    env.reset()

