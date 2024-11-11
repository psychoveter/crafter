#%%
import numpy as np
from numpy._typing import ArrayLike

import crafter
from mv.draw_utils import render_nparr_onehot, draw_image_grid
from mv.utils import create_nparr_onehot, sample_nparr_onehot, miss_rate_all
from PIL import Image
import matplotlib.pyplot as plt
from mv.const import objects

#%%
env = crafter.Env()
env.reset()

obs, reward, done, info = env.step(env.action_space.sample())

arr = create_nparr_onehot(env)
img = render_nparr_onehot(arr, env)

img = draw_image_grid([img] * 3)
Image.fromarray(img).show()


#%%
def crop_field_from_obs(obs):
    """
    Get game field from the observation field
    :param obs: img 64 x 64 x 3
    :return: image 63 * 49 * 3 with field view
    """


    pass

#%%
env = crafter.Env()



imgs = sample_nparr_onehot(5, env)
line = draw_image_grid([render_nparr_onehot(img, env) for img in imgs])
Image.fromarray(line).show()




print(miss_rate_all(imgs[:4], imgs[1:]))

#%%
import random

env = crafter.Env()
env.reset()
pos = [
    random.randint(4, env._size[0] - 4),
    random.randint(4, env._size[1] - 4)
]
print(pos)

npr = create_nparr_onehot(env, pos=pos)
print(npr)
img = render_nparr_onehot(npr, env)
Image.fromarray(img).show()


#%% MSE loss amount
import torch
npr2 = np.copy(npr)
npr2[2][0][0] = 1 if npr[2][0][0] == 0 else 1
npr2[2][1][0] = 1 if npr[2][1][0] == 0 else 1
npr2[2][2][0] = 1 if npr[2][2][0] == 0 else 1

t1 = torch.tensor([npr] * 10, dtype=torch.float32)
t2 = torch.tensor([npr] * 10, dtype=torch.float32)
print(t1)
loss = torch.nn.CrossEntropyLoss()
l1 = loss(t1, t2)
print(l1.item())

#%%
import torch
q1 = torch.zeros((10, 6 , 6), dtype=torch.float32)
q2 = torch.zeros((10, 6 , 6), dtype=torch.float32)
q1[1,:,:] = 1
q2[2,:,:] = 1
print(q1.sum())

cel = torch.nn.CrossEntropyLoss()
with torch.no_grad():
    q1 = q1.transpose(2, 0)
    q2 = q2.transpose(2,0)
    print(f"CE: {cel(q1, q2).item()}")

mse = torch.nn.MSELoss()
print(f"MSE: {mse(q1, q2).item()}")


#%%
np.savez('sample.npz', npr=npr)
nprr = np.load('sample.npz')['npr']
eq = npr == nprr

#%%
last = npr
env.reset()
for i in range(10):
    env.step(env.action_space.sample())
    npr = create_nparr_onehot(env)
    print(npr.sum())


#%%
lis = [1,2,3,4,5,6,7]

print(max(lis))