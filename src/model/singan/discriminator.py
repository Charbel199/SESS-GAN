# Code based on https://github.com/tamarott/SinGAN
import torch
import torch.nn as nn
from model.singan.conv_block import ConvBlock
from conf import ModelConfig


class Discriminator(nn.Module):
    """ Patch based Discriminator. Uses Namespace opt. """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(config.number_of_filters)
        self.head = ConvBlock(len(config.token_list), N, (config.kernel_size, config.kernel_size), 0,
                              1)  # Padding is done externally
        self.body = nn.Sequential()

        for i in range(config.number_of_layers - 2):
            block = ConvBlock(N, N, (config.kernel_size, config.kernel_size), 0, 1)
            self.body.add_module("block%d" % (i + 1), block)

        block = ConvBlock(N, N, (config.kernel_size, config.kernel_size), 0, 1)
        self.body.add_module("block%d" % (config.number_of_layers - 2), block)

        self.tail = nn.Conv2d(N, 1, kernel_size=(config.kernel_size, config.kernel_size), stride=1, padding=0)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x
