from uu import encode

from mv.model.autoencoder import CrafterEnvEncoder2dV0
import torch


x = torch.randn(32, 64, 16)



c1 = torch.nn.ConvTranspose1d(64, 16, kernel_size=5, padding=2, stride=1, output_padding=0)

y = c1(x)
print(y.shape)