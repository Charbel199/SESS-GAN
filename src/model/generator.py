import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, noise_dimension, number_of_classes, features_g, kernel_size):
        super(Generator, self).__init__()
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
