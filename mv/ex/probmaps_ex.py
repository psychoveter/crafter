
import numpy as np
import PIL.Image as Image

import crafter
from mv.autoencoder import load_model
from mv.utils import create_tensor_onehot, render_tensor_onehot, draw_image_grid

env = crafter.Env()
env.reset()

run_folder = '/Users/Oleg.Bukhvalov/projects/montevideo/crafter/mv/ray_results/autoencoder-0/TorchTrainer_dcc41_00000_0_2024-10-29_12-40-41'
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

def channel_to_img(tensor, channel, side_size=32):
    b, c, w, h = tensor.shape
    arr = tensor[0, channel].detach().numpy()
    arr *= 127
    arr = arr.astype(np.uint8)

    img = np.zeros((side_size * w, side_size * h,3), np.uint8)
    for i in range(w):
        for j in range(h):
            img[i*side_size:(i+1)*side_size, j*side_size:(j+1)*side_size, 0] = arr[i,j]


    return img.transpose(1,0,2)


images = [
    source_img,
    target_img,
    channel_to_img(sample, 1),
    channel_to_img(sample, 2),
    channel_to_img(sample, 6),
    channel_to_img(sample, 19)
]

image_line = draw_image_grid(images)
Image.fromarray(image_line).show()