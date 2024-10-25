#%%
import numpy as np

import crafter
from PIL import Image

env = crafter.Env()
env.reset()

obs, reward, done, info = env.step(env.action_space.sample())

# img = Image.fromarray(obs)
# img.show()

#%%
from mv.utils import objects, get_object_dict
print(get_object_dict(None))

#%%
from mv.utils import create_tensor_onehot

# translate into one hot matrix
npr = create_tensor_onehot(env)
print(npr.shape)

grass = npr[2,:,:]
print(npr.shape)
print(grass)

#%% MSE loss amount
import torch
npr2 = np.copy(npr)
# npr2[2][0][0] = 1 if npr[2][0][0] == 0 else 1
# npr2[2][1][0] = 1 if npr[2][1][0] == 0 else 1
# npr2[2][2][0] = 1 if npr[2][2][0] == 0 else 1

t1 = torch.tensor([npr] * 10, dtype=torch.float32)
t2 = torch.tensor([npr2] * 10, dtype=torch.float32)
print(t1)
loss = torch.nn.CrossEntropyLoss()
l1 = loss(t1, t2)
print(l1.item())

#%%
np.savez('sample.npz', npr=npr)
nprr = np.load('sample.npz')['npr']
eq = npr == nprr

#%%
last = npr
env.reset()
for i in range(10):
    env.step(env.action_space.sample())
    npr = create_tensor_onehot(env)
    print(npr.sum())


#%%
