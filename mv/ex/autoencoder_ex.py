#%%
from typing import Union, Callable, Tuple, Any, Optional, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

import crafter
from mv.utils import create_nparr_onehot
from mv.const import objects, object_weights
from mv.autoencoder import create_datasets, CrafterAutoencoderEnv2dV0, CrafterDatasetEnv2d
import torch.nn.functional as F

print(f"torch=={torch.__version__}")
print(f"np=={np.__version__}")


train_set, test_set = create_datasets(100, 100)
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
test_loader = DataLoader(test_set, batch_size=16, shuffle=True)

model = CrafterAutoencoderEnv2dV0(
    channels_size=[32, 64, 64, 64, 32],
    latent_size=32
)
sample = train_loader.__iter__().__next__()
res = model(sample)
print(res.shape)

# l = crafter_onehot_loss(res, sample)

#%%
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
loss_fun = torch.nn.CrossEntropyLoss()

losses = []
for epoch in range(200):
    for batch in train_loader:
        optimizer.zero_grad()
        output = model(batch)
        loss = loss_fun(output, batch)
        loss.backward()
        optimizer.step()
    print(f"Epoch: {epoch}, Loss: {loss.item()}")
    losses.append(loss.item())

#%%
import matplotlib.pyplot as plt
plt.plot(range(len(losses)), losses)
plt.show()
