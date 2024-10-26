import os
import tempfile

import ray
import ray.train
import ray.tune
import ray.train.torch
from ray.train import ScalingConfig, RunConfig
from ray.train.torch import TorchConfig, TorchTrainer

import torch
from torch.utils.data import DataLoader
from mv.autoencoder import CrafterEnvAutoencoderV0, CrafterEnvDataset, create_datasets, create_autoencoder
from mv.utils import get_actual_device, object_weights

torch_object_weights = torch.tensor(object_weights, dtype=torch.float32) #+ (torch.rand(len(object_weights)))

def crafter_onehot_loss(input, output):
    """
    Computes loss for crafter representation scene using object weights from mv.utils.objects dict.
    This weighting aims to be more precise about objects and rare materials, than usual
    :param input: tensor of shape BS W H C
    :param output: tensor of shape BS W H C
    :return: float error value
    """
    batch_size, width, height, channels = input.shape
    print(f"input shape: {input.shape}, output shape: {output.shape}")
    diff = input - output
    diff = (torch.pow(diff, 2))

    # norm = diff.norm()
    # diff = diff / norm
    # diff[:,:,:] *= torch_object_weights

    return diff.sum()

def execute_sample_torch():
    print(f"Torch version is {torch.__version__}")
    print(f"Cuda is available: {torch.cuda.is_available()}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    v1 = torch.randn((2, 2), dtype=torch.float32, device=device)
    v2 = torch.randn((2, 2), dtype=torch.float32, device=device)

    v3 = v1 + v2
    print(v3)
    print(f"v3 device: {v3.device}")


def tune_autoencoder(config):
    train_autoencoder(config, is_ray_train = False)

def train_autoencoder(config, is_ray_train = True):
    print(f"Start train_autoencoder loop at cluster {ray.cluster_resources()}, "
          f"torch version is {torch.__version__}, config is {config}")

    # get parameters
    learning_rate = config['learning_rate']
    batch_size = int(config['batch_size'])
    dataset_size = int(config['dataset_size'])
    max_epochs = int(config['max_epochs'])

    # generate data
    train_set, test_set = create_datasets(dataset_size, int(dataset_size / 10))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    # create model
    device = get_actual_device()
    # device = torch.device("cpu")
    print(f"Device is {device}")
    model = create_autoencoder(config)
    model.to(device)

    # add hooks
    def backward_hook(module: torch.nn.Module, grad_input, grad_output):
        print(f"backward hook: {module._get_name()}, input_norm: {torch.norm(grad_input[0])}, output_norm: {torch.norm(grad_output[0])}")
    model.register_backward_hook(backward_hook)

    # prepare training
    if is_ray_train:
        ray.train.torch.prepare_model(model, move_to_device=False)
        ray.train.torch.prepare_data_loader(train_loader, move_to_device=False)


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fun = torch.nn.CrossEntropyLoss()
    # loss_fun = crafter_onehot_loss

    losses = []
    for epoch in range(max_epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            batch = batch.to(device)
            output = model(batch)
            # output = output.to(device)
            loss = loss_fun(output, batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")
        losses.append(loss.item())

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            torch.save(model.state_dict(), os.path.join(temp_checkpoint_dir, "model.pt"))
            metrics = {"loss": loss.item()}  # Training/validation metrics.
            checkpoint = ray.train.Checkpoint.from_directory(temp_checkpoint_dir)
            ray.train.report(metrics=metrics, checkpoint=checkpoint)

    return max(losses)


def run_torch_train():
    config = {
            'batch_size': 32,
            'dataset_size': 1000,
            'dropout': 0.3,
            'hidden_channel_0': 64,
            'hidden_channel_1': 64,
            'hidden_channel_2': 64,
            'hidden_channel_3': 32,
            'hidden_channel_4': 32,
            'latent_size': 64,
            'learning_rate': 0.01,
            'max_epochs': 500
    }

    # scaling_config = ScalingConfig(num_workers=1, use_gpu=True)
    run_config = RunConfig(storage_path="~/projects/montevideo/crafter/mv/ray_results", name="autoencoder-0")

    trainer = TorchTrainer(
        train_autoencoder,
        train_loop_config=config,
        # scaling_config=scaling_config,
        run_config=run_config)

    trainer.fit()

if __name__ == "__main__":
    run_torch_train()