#%%

import crafter
from mv.model.autoencoder import load_model
from mv.utils import create_tensor_onehot
from mv.const import object_keys
from mv.draw_utils import render_tensor_onehot, plot_image_grid, channel_to_img, render_channels
import PIL.Image as Image

env = crafter.Env()
env.reset()

run_folder = '/Users/Oleg.Bukhvalov/projects/montevideo/crafter/mv/ray_results/autoencoder-0/TorchTrainer_c7c66_00000_0_2024-10-29_18-09-23'
# run_folder = '/Users/Oleg.Bukhvalov/projects/montevideo/crafter/mv/ray_results/autoencoder-0/TorchTrainer_ee126_00000_0_2024-10-29_10-46-38'

print(f"run_folder: {run_folder}")
model = load_model(run_folder)
model.eval()

input = create_tensor_onehot(env)
sample = model(input)
#B C W H
print(sample.shape)

source_img = render_tensor_onehot(input[0], env)
target_img = render_tensor_onehot(sample[0], env)
# B 3 W H

#%%

images = [
    ("source", source_img),
    ("target", target_img),
    (object_keys[1], channel_to_img(sample, 1)),
    (object_keys[2], channel_to_img(sample, 2)),
    (object_keys[6], channel_to_img(sample, 6)),
    (object_keys[13], channel_to_img(sample, 13)),
    (object_keys[14], channel_to_img(sample, 14)),
    (object_keys[15], channel_to_img(sample, 15)),
    (object_keys[16], channel_to_img(sample, 16)),
    (object_keys[17], channel_to_img(sample, 17)),
    (object_keys[18], channel_to_img(sample, 18)),
    (object_keys[19], channel_to_img(sample, 19)),
    (object_keys[20], channel_to_img(sample, 20)),
    (object_keys[21], channel_to_img(sample, 21)),
]

# image_line = draw_image_grid(images)
# Image.fromarray(image_line).show()

plot_image_grid(
    images = [i[1] for i in images],
    titles = [i[0] for i in images],
    grid_size = (3, len(images) // 3 + 1),
    figsize = (12,12)
)

Image.fromarray(render_channels(sample[0], 5, 5)).show()
print(sample[0,:,5,5])