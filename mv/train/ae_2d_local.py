#%%
import ray

ray.init()

#%%
@ray.remote
def some_task(item):
    print(f"Received task {item}")


#%%
import torch
print(torch.cuda.is_available())
print(torch.backends.mps.is_available())