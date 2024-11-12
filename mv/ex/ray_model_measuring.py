#%%
import os.path

import PIL.Image as Image
import crafter
from mv.model.autoencoder import create_autoencoder_2d, load_tune_run
from mv.utils import render_nparr_onehot, draw_image_grid, create_tensor_onehot

results_path     = '/Users/Oleg.Bukhvalov/projects/montevideo/crafter/mv/ray_results'
tune_folder_path = 'tune_autoencoder_2024-10-25_16-31-11'
run_folder_name  = 'tune_autoencoder_6678e_02796_2796_batch_size=16,dataset_size=500,dropout=0.3389,hidden_channel_0=64,hidden_channel_1=64,hidden_cha_2024-10-25_20-21-45'

tune_folder = os.path.join(results_path, tune_folder_path)
run_folder = os.path.join(tune_folder, run_folder_name)


def sample_and_show(model, env):
    """
    Creates env and samples an image from the env, then project it via the model
    :param model:
    :return:
    """

    env.step(env.action_space.sample())
    torch_sample = create_tensor_onehot(env)

    # print(torch_sample.shape)
    restored = model(torch_sample).detach().numpy().round()
    # print(f"restored: {restored.shape}")


    img_source = render_nparr_onehot(torch_sample[0].detach().numpy(), env)
    img_restored = render_nparr_onehot(restored[0], env)
    img = draw_image_grid([img_source, img_restored])

    Image.fromarray(img).show()

run_folder = '/Users/Oleg.Bukhvalov/projects/montevideo/crafter/mv/ray_results/autoencoder-0/TorchTrainer_c8284_00000_0_2024-10-28_13-24-15'
# checkpoint = 'checkpoint_0000120'
checkpoint=None
model_state, params = load_tune_run(run_folder, checkpoint=checkpoint)
print(f"Params for the run {params}")
model = create_autoencoder_2d(params)
model.load_state_dict(model_state)

env = crafter.Env()
env.reset()
sample_and_show(model, env)

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