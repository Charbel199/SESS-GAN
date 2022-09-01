import torch.nn as nn
import torch
from matplotlib import pyplot as plt


class Generator(nn.Module):
    def __init__(self, noise_dimension, number_of_classes, features_g, kernel_size, device):
        self.noise_dimension = noise_dimension
        self.number_of_classes = number_of_classes
        self.features_g = features_g
        self.kernel_size = kernel_size
        self.device = device
        super(Generator, self).__init__()
        self.kwargs = {'noise_dimension': noise_dimension,
                       'number_of_classes': number_of_classes,
                       'features_g': features_g,
                       'kernel_size': kernel_size,
                       'device': device}
        # Based on  DCGAN paper
        self.network = nn.Sequential(
            self._block(noise_dimension, features_g * 16, kernel_size, 1, 0),
            self._block(features_g * 16, features_g * 8, kernel_size, 2, 1),
            self._block(features_g * 8, features_g * 4, kernel_size, 2, 1),
            self._block(features_g * 4, features_g * 2, kernel_size, 2, 1),
            nn.ConvTranspose2d(features_g * 2, number_of_classes, kernel_size=2, stride=2, padding=1),
            nn.Softmax(dim=1)  # Set probability for each class
        )

    @staticmethod
    def _block(in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            # Bias false for batch norm
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.network(x)

    def generate_image(self, show_environment=False):
        noise = torch.randn(1, self.noise_dimension, 1, 1).to(self.device)
        generated_environment = self.forward(noise)
        generated_environment = generated_environment[0]
        generated_environment = torch.argmax(generated_environment, dim=0)
        if show_environment:
            plt.rcParams["figure.figsize"] = [7.00, 3.50]
            plt.rcParams["figure.autolayout"] = True
            im = plt.imshow(generated_environment.cpu(), cmap="jet")
            plt.colorbar(im)
            plt.show()
        return generated_environment
