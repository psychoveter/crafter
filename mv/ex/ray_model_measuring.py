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

results_path     = '/Users/Oleg.Bukhvalov/projects/montevideo/crafter/mv/ray_results'
tune_folder_path = 'tune_autoencoder_2024-10-25_16-31-11'
run_folder_name  = 'tune_autoencoder_6678e_02796_2796_batch_size=16,dataset_size=500,dropout=0.3389,hidden_channel_0=64,hidden_channel_1=64,hidden_cha_2024-10-25_20-21-45'

tune_folder = os.path.join(results_path, tune_folder_path)
run_folder = os.path.join(tune_folder, run_folder_name)

def load_tune_run(run_folder):

    progress = pd.read_csv(os.path.join(run_folder, "progress.csv"))
    # load best loss checkpoint
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


def sample_and_show(model):
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

run_folder = '/Users/Oleg.Bukhvalov/projects/montevideo/crafter/mv/ray_results/autoencoder-0/TorchTrainer_c8284_00000_0_2024-10-28_13-24-15'
model_state, params = load_tune_run(run_folder)
print(f"Params for the run {params}")
model = create_autoencoder(params)
model.load_state_dict(model_state)

sample_and_show(model)

# def accuracy(model: CrafterEnvAutoencoderV0, num_samples: int):
#     env = crafter.Env()
#     env.reset()
#

"""
Best run with cross entropy loss
Result(
  metrics={'loss': 2.445498466491699},
  path='/Users/Oleg.Bukhvalov/projects/montevideo/crafter/mv/ray_results/tune_autoencoder_2024-10-25_16-31-11/tune_autoencoder_6678e_02796_2796_batch_size=16,dataset_size=500,dropout=0.3389,hidden_channel_0=64,hidden_channel_1=64,hidden_cha_2024-10-25_20-21-45',
  filesystem='local',
  checkpoint=Checkpoint(filesystem=local, path=/Users/Oleg.Bukhvalov/projects/montevideo/crafter/mv/ray_results/tune_autoencoder_2024-10-25_16-31-11/tune_autoencoder_6678e_02796_2796_batch_size=16,dataset_size=500,dropout=0.3389,hidden_channel_0=64,hidden_channel_1=64,hidden_cha_2024-10-25_20-21-45/checkpoint_000099)
)
"""