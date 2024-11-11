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
from mv.autoencoder import CrafterAutoencoderEnv2dV0, CrafterDatasetEnv2d, create_datasets
from mv.ray.example_pytorch import tune_autoencoder

search_space = {
    'learning_rate': tune.uniform(0.0001, 0.01),
    'batch_size': tune.grid_search([256, 512, 124]),
    'hidden_channel_0': tune.grid_search([32, 64]),
    'hidden_channel_1': tune.grid_search([32, 64, 128]),
    'hidden_channel_2': tune.grid_search([32, 64, 96]),
    'hidden_channel_3': tune.grid_search([32, 64]),
    'hidden_channel_4': tune.grid_search([16, 32]),
    'latent_size': tune.uniform(16, 128),
    'encoder_dropout': tune.uniform(0.1, 0.5),
    'decoder_dropout': tune.uniform(0.05, 0.15),
    'dataset_size': tune.grid_search([15000]),
    'max_epochs': tune.grid_search([1000])
}

def setup_tuner(search_space: dict[str, Any]) -> Tuple[TrialScheduler, Tuner]:
    scheduler = tune.schedulers.ASHAScheduler(
        max_t=1000,
        metric="loss",
        mode="min",
        grace_period=2,
    )

    tuner = tune.Tuner(
        trainable=tune.with_resources(
            tune.with_parameters(tune_autoencoder),
            resources={"cpu": 1, "gpu": 0.5}
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

# if __name__ == "__main__":
#     ray.init()
#     run_tune_train()
    # load_tuner()

run_tune_train()

"""
TODO: 
 - better flow of running jobs on cluster
    - how to run job from single script
        -- git commit -- git push -- remote run get pull && checkout mv-crafter && python run_tune
    - run tensorboard and forward ports to local 
    
 - install dependencies (like opensimplex)
"""