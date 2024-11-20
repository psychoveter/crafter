import math
import os
import tempfile

import ray
import torch
from ray.air import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
import crafter
from mv.datagen import CrafterAgentDataset3D, create_dataloader_3d
from mv.model.autoencoder import load_model as load_autoencoder_2d
from mv.model.autoencoder2plus1 import Autoencoder2plus1, load_ae_2plus1
from mv.train.ae2d_train import partite_sigmoid_focal_loss
from mv.utils import get_actual_device


def film_partite_sigmoid_focal_loss(inputs, targets, object_alpha=0.7):
    b, f, c, h, w = inputs.shape
    inputs = inputs.view(-1, c, h, w)
    targets = targets.view(-1, c, h, w)
    loss = partite_sigmoid_focal_loss(inputs, targets, object_alpha=object_alpha)
    return loss


def train_ae_2plus1(config):

    device = get_actual_device()
    model = load_ae_2plus1(
        ae2d_folder=config['ae2d_folder'],
        film_length=int(config['film_length']),
        latent_size_3d=int(config['latent_size_3d']),
    )

    model.to(device)

    data_loader = create_dataloader_3d(
        int(config['dataset_size']),
        int(config['film_length']),
        int(config['batch_size'])
    )

    def loss_fun_alpha(inputs, targets):
        return film_partite_sigmoid_focal_loss(inputs, targets, object_alpha=config['object_alpha'])
    loss_fun = loss_fun_alpha

    def backward_hook(module: torch.nn.Module, grad_input, grad_output):
        print(f"backward hook: {module._get_name()}, input_norm: {torch.norm(grad_input[0])}, output_norm: {torch.norm(grad_output[0])}")
    model.encoder1d.register_backward_hook(backward_hook)
    model.decoder1d.register_backward_hook(backward_hook)

    from itertools import chain
    # params = chain(model.encoder1d.parameters(), model.decoder1d.parameters())
    params = model.parameters()
    optimizer = torch.optim.AdamW(params, lr=float(config['learning_rate']))
    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, int(config['max_epochs']))

    losses = []
    for epoch in range(int(config['max_epochs'])):
        for batch in data_loader:
            optimizer.zero_grad()
            batch = batch.to(device)
            print(f"batch shape: {batch.shape}")
            output = model(batch)
            loss = loss_fun(output, batch)
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                lr_scheduler.step()

        print(f"Epoch: {epoch}, Loss: {loss.item()}")
        losses.append(loss.item())

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            torch.save(model.state_dict(), os.path.join(temp_checkpoint_dir, "model.pt"))
            metrics = {"loss": loss.item()}  # Training/validation metrics.
            checkpoint = ray.train.Checkpoint.from_directory(temp_checkpoint_dir)
            ray.train.report(metrics=metrics, checkpoint=checkpoint)

    return max(losses)

def run_ae_2plus1_train():
    # 'batch_size': 64,
    # 'dataset_size': 1000,
    # 'film_length': 8,
    # 'latent_size_3d': 512,
    # 'learning_rate': 0.0001,
    # 'max_epochs': 1000
    config = {
        'ae2d_folder': f"{os.path.expanduser('~')}/projects/montevideo/crafter/mv/ray_results/ae2d-0/TorchTrainer_7d0b1_00000_0_2024-11-20_05-43-56",
        'batch_size': 512,
        'dataset_size': 15000,
        'film_length': 16,
        'latent_size_3d': 1024,
        'learning_rate': 0.001 * math.sqrt(512 / 128),
        'max_epochs': 1000,
        'object_alpha': 0.7,
    }


    run_config = RunConfig(storage_path="~/projects/montevideo/crafter/mv/ray_results", name="ae-2plus1-0")
    scaling_config = ScalingConfig(num_workers=1, use_gpu=True)
    trainer = TorchTrainer(
        train_ae_2plus1,
        train_loop_config=config,
        scaling_config=scaling_config,
        run_config=run_config
    )

    trainer.fit()

if __name__ == "__main__":
    run_ae_2plus1_train()