import os
import tempfile
from typing import Tuple, Any

import ray
import ray.train
import ray.tune as tune
import ray.train.torch
from ray.tune import Tuner
from ray.tune.schedulers import TrialScheduler
from ray.train import ScalingConfig, RunConfig
from ray.train.torch import TorchConfig, TorchTrainer
import torch
from torch.utils.data import DataLoader
from mv.autoencoder import CrafterEnvAutoencoderV0, CrafterEnvDataset, create_datasets
from mv.ray.example_pytorch import tune_autoencoder

search_space = {
    'learning_rate': tune.uniform(0.0001, 0.1),
    'batch_size': tune.grid_search([16, 32]),
    'hidden_channel_0': tune.grid_search([32, 64]),
    'hidden_channel_1': tune.grid_search([32, 64, 128]),
    'hidden_channel_2': tune.grid_search([32, 64]),
    'hidden_channel_3': tune.grid_search([32, 64]),
    'hidden_channel_4': tune.grid_search([16, 32]),
    'latent_size': tune.uniform(4, 16),
    'dropout': tune.uniform(0.2, 0.5),
    'dataset_size': tune.grid_search([500, 1000]),
    'max_epochs': tune.grid_search([200, 300])
}

def setup_tuner(search_space: dict[str, Any]) -> Tuple[TrialScheduler, Tuner]:
    scheduler = tune.schedulers.ASHAScheduler(
        max_t=100,
        metric="loss",
        mode="min",
        grace_period=2,
    )

    tuner = tune.Tuner(
        trainable=tune.with_resources(
            tune.with_parameters(tune_autoencoder),
            resources={"cpu": 1}
        ),
        param_space=search_space,
        run_config=RunConfig(storage_path="~/projects/montevideo/crafter/mv/ray_results"),
        tune_config=tune.TuneConfig(
            num_samples=10,
            scheduler=scheduler,
        ),
    )
    return scheduler, tuner

def run_tune_train():
    scheduler, tuner = setup_tuner(search_space)
    results = tuner.fit()
    best_result = results.get_best_result("loss", "min")
    print(best_result)

def load_tuner():
    tuner = tune.Tuner.restore(
        path="/Users/Oleg.Bukhvalov/projects/montevideo/crafter/mv/ray_results/train_autoencoder_2024-10-23_19-45-28",
        trainable=tune.with_resources(
            tune.with_parameters(tune_autoencoder),
            resources={"cpu": 1}
        )
    )
    results = tuner.fit()
    best_result = results.get_best_result("loss", "min")
    print(best_result)
    print(best_result.config)

if __name__ == "__main__":
    ray.init()
    run_tune_train()
    # load_tuner()