import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, number_of_classes, features_d):
        # Features_d is just used for simpler scaling of the convolution neural network
        super(Discriminator, self).__init__()
        # Based on  DCGAN paper
        self.network = nn.Sequential(
            nn.Conv2d(number_of_classes, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid()
        )

    @staticmethod
    def _block(in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),  # Bias false for batch norm
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.network(x)
