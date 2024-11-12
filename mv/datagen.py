import math
import random
from typing import Tuple
from venv import create

import numpy as np
import torch
from stable_baselines3 import PPO
from torch.utils.data import Dataset, DataLoader

import crafter
from mv.draw_utils import render_film_np_onehot
from mv.utils import sample_nparr_onehot, create_nparr_onehot, exec_and_measure


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

    def __init__(self, env, model,
                 dataset_size,
                 film_length,
                 max_films_per_run = 20,
                 ):
        super(CrafterAgentDataset3D, self).__init__()
        self.env = env
        self.model = model
        self.film_length = film_length

        np_films = sample_np_films(env, model,
                                film_length,
                                runs = int(1 + dataset_size / max_films_per_run),
                                max_films = dataset_size,
                                max_films_per_run = max_films_per_run)

        self.films = [torch.Tensor(npf) for npf in np_films]


    def __len__(self):
        return len(self.films)

    def __getitem__(self, idx):
        return self.films[idx]


def sample_np_films(env, model,
                    film_length,
                    runs: int = 1,
                    max_films: int = None,
                    max_films_per_run: int = None):

    films = []
    runs_done = 0
    for i in range(runs):
        env.reset()
        action = env.action_space.sample()
        state, done, onehot_frames = (None, False, [])
        while not done:
            obs, reward, done, info = env.step(action)
            action, state = model.predict(obs, state)
            np_onehot = create_nparr_onehot(env)
            onehot_frames.append(np_onehot)

        print(f"Run {i}, done {len(onehot_frames)} frames")
        if len(onehot_frames) >= film_length:
            start_indexes = list(range(len(onehot_frames) - film_length + 1))
            indexes_to_skip = max(len(start_indexes) - max_films_per_run, 0)
            for i in range(indexes_to_skip):
                to_skip = random.randint(0, len(start_indexes) - 1)
                start_indexes.pop(to_skip)
            for si in start_indexes:
                film = onehot_frames[si:si + film_length]
                film = np.array(film)
                films.append(film)
                if max_films and len(films) >= max_films:
                    return films
        runs_done += 1
    return films


def create_film(env, model, film_length):
    step = 0
    env.reset()
    action = env.action_space.sample()
    state, done, onehot_frames = (None, False, [])
    while not done and step < film_length:
        step += 1
        obs, reward, done, info = env.step(action)
        action, state = model.predict(obs, state)
        np_onehot = create_nparr_onehot(env)
        onehot_frames.append(np_onehot)

    if len(onehot_frames) == film_length:
        np_frames = np.array(onehot_frames)
        t_onehot = torch.Tensor(np_frames)
        return t_onehot
    else:
        return None

def create_dataloader_3d(dataset_size, film_length, batch_size):
    env = crafter.Env()
    root = '/Users/Oleg.Bukhvalov/projects/montevideo/crafter'
    model = PPO.load(f"{root}/ppo.zip")

    dataset = CrafterAgentDataset3D(env, model, dataset_size, film_length)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return data_loader




if __name__ == '__main__':
    env = crafter.Env()
    root = '/Users/Oleg.Bukhvalov/projects/montevideo/crafter'
    model = PPO.load(f"{root}/ppo.zip")
    # films, elapsed = exec_and_measure(lambda: sample_np_films(env, model,
    #                                                           runs=2,
    #                                                           max_films=20,
    #                                                           film_length=64,
    #                                                           max_films_per_run=10))
    # print(len(films))
    # print(films[0].shape)
    # print(f"Total time {elapsed}, film in {elapsed / len(films)}")
    # render_film_np_onehot(films[0], env)

    dataset, elapsed = exec_and_measure(lambda: CrafterAgentDataset3D(env, model, 100, 64))
    print(len(dataset))
    print(dataset[0].shape)
    print(f"Total time {elapsed}, film in {elapsed / len(dataset)}")
    render_film_np_onehot(dataset[0], env)