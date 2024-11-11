from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

import crafter
from mv.utils import sample_nparr_onehot, create_nparr_onehot


class CrafterDatasetEnv2d(Dataset):
    def __init__(self, env, size, samples_from_world: int = 1000):
        self.env = env
        self.env.reset()

        self.items = []

        samples = sample_nparr_onehot(size, env, samples_from_world=samples_from_world)
        for sample in samples:
            sample = torch.tensor(sample, dtype=torch.float32)
            sample = sample.contiguous()
            self.items.append(sample)


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def create_datasets_2d(train_size: int, test_size: int) -> Tuple[CrafterDatasetEnv2d, CrafterDatasetEnv2d]:
    env = crafter.Env()
    train_set = CrafterDatasetEnv2d(env, size=train_size)
    test_set = CrafterDatasetEnv2d(env, size=test_size)
    return train_set, test_set


class CrafterAgentDataset3D(Dataset):
    def __init__(self, env, model, dataset_size, film_length):
        super(CrafterAgentDataset3D, self).__init__()
        self.env = env
        self.model = model
        self.film_length = film_length
        self.films = []

        self.env.reset()
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
