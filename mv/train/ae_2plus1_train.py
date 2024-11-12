import os
import tempfile

import ray
import torch
from ray.air import RunConfig
from ray.train.torch import TorchTrainer
import crafter
from mv.datagen import CrafterAgentDataset3D, create_dataloader_3d
from mv.model.autoencoder2plus1 import Autoencoder2plus1
from mv.train.ae2d_train import partite_sigmoid_focal_loss
from mv.utils import get_actual_device



def film_partite_sigmoid_focal_loss(inputs, targets):
    loss = torch.vmap(func=partite_sigmoid_focal_loss, in_dims=1)
    loss = loss(inputs, targets)
    loss = loss.sum()
    return loss


def train_ae_2plus1(config):

    device = get_actual_device()
    model = Autoencoder2plus1(
        film_length=int(config['film_length']),
        latent_size_2d=int(config['latent_size_2d']),
        latent_size_3d=int(config['latent_size_3d']),
        channels_size_2d=[
            int(config['encoder_2d_hidden_channel_0']),
            int(config['encoder_2d_hidden_channel_1']),
            int(config['encoder_2d_hidden_channel_2']),
            int(config['encoder_2d_hidden_channel_3']),
            int(config['encoder_2d_hidden_channel_4']),
        ]
    )
    model.to(device)

    data_loader = create_dataloader_3d(
        int(config['dataset_size']),
        int(config['film_length']),
        int(config['batch_size'])
    )

    loss_fun = film_partite_sigmoid_focal_loss

    def backward_hook(module: torch.nn.Module, grad_input, grad_output):
        print(f"backward hook: {module._get_name()}, input_norm: {torch.norm(grad_input[0])}, output_norm: {torch.norm(grad_output[0])}")
    model.register_backward_hook(backward_hook)

    from itertools import chain
    params = chain(model.encoder1d.parameters(), model.decoder1d.parameters())
    optimizer = torch.optim.AdamW(params, lr=float(config['learning_rate']))
    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, int(config['max_epochs']))

    losses = []
    for epoch in range(int(config['max_epochs'])):
        for batch in data_loader:
            optimizer.zero_grad()
            batch = batch.to(device)
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
    config = {
        'batch_size': 32,
        'dataset_size': 500,
        'film_length': 32,
        'encoder_2d_dropout': 0.3,
        'decoder_2d_dropout': 0.1,
        'encoder_2d_hidden_channel_0': 64,
        'encoder_2d_hidden_channel_1': 96,
        'encoder_2d_hidden_channel_2': 96,
        'encoder_2d_hidden_channel_3': 64,
        'encoder_2d_hidden_channel_4': 32,
        'latent_size_2d': 64,
        'latent_size_3d': 256,
        'learning_rate': 0.001,
        'max_epochs': 1000
    }

    run_config = RunConfig(storage_path="~/projects/montevideo/crafter/mv/ray_results", name="ae-2plus1-0")
    trainer = TorchTrainer(
        train_ae_2plus1,
        train_loop_config=config,
        run_config=run_config
    )

    trainer.fit()

if __name__ == "__main__":
    run_ae_2plus1_train()