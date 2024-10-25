#%%
import numpy as np
from numpy._typing import ArrayLike

import crafter
from mv.utils import create_nparr_onehot, render_nparr_onehot, draw_image_grid
from PIL import Image
import matplotlib.pyplot as plt
from mv.utils import objects, get_object_dict

#%%
env = crafter.Env()
env.reset()

obs, reward, done, info = env.step(env.action_space.sample())

arr = create_nparr_onehot(env)
img = render_nparr_onehot(arr, env)

img = draw_image_grid([img] * 3)
Image.fromarray(img).show()

#%%
env = crafter.Env()



imgs = sample_nparr_onehot(5, env)
line = draw_image_grid([render_nparr_onehot(img, env) for img in imgs])
Image.fromarray(line).show()




print(miss_rate_all(imgs[:4], imgs[1:]))

#%%
# translate into one hot matrix
npr = create_nparr_onehot(env)
print(npr.shape)

grass = npr[2,:,:]
print(npr.shape)
print(grass)

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
q1 = torch.zeros((10, 6 , 6), dtype=torch.float32)
q2 = torch.zeros((10, 6 , 6), dtype=torch.float32)
q1[1,:,:] = 1
q2[1,:,:] = 1

cel = torch.nn.CrossEntropyLoss()
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