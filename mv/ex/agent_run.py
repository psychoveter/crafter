#%%
import numpy as np
import torch

import crafter
from stable_baselines3 import PPO

root = '/Users/Oleg.Bukhvalov/projects/montevideo/crafter'

model = PPO.load(f"{root}/ppo.zip")

env = crafter.Env()
# env = crafter.Recorder(env=env, directory=f"{root}/mv/agent_results")
env.reset()

#%%
action = env.action_space.sample()
state = None
done = False
while not done:
    obs, reward, done, info = env.step(action)
    print(info)
    action, state = model.predict(obs, state)
    print(f"action selected: {action}")



#%%
from torch.utils.data import Dataset
from mv.utils import create_nparr_onehot, create_tensor_onehot
import numpy as np

class CrafterAgentDataset(Dataset):
    def __init__(self, env, model, dataset_size, film_length):
        super(CrafterAgentDataset, self).__init__()
        self.env = env
        self.model = model
        self.film_length = film_length
        self.films = []

        for i in range(dataset_size):
            step = 0
            action = env.action_space.sample()
            state = None
            done = False

            onehot_frames = []
            while not done and step < self.film_length:
                step += 1
                obs, reward, done, info = env.step(action)
                action, state = model.predict(obs, state)
                np_onehot = create_nparr_onehot(env)
                onehot_frames.append(np_onehot)

            np_frames = np.array(onehot_frames)
            t_onehot = torch.Tensor(np_frames)
            self.films.append(t_onehot)

        print(f"created {len(np_frames)} frames")

    def __len__(self):
        return len(self.films)

    def __getitem__(self, idx):
        return self.films[idx]

cad = CrafterAgentDataset(env, model, 1, 100)
sample = cad.__getitem__(0)