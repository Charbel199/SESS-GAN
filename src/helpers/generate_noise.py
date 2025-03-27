# Code based on https://github.com/tamarott/SinGAN
import torch


def generate_spatial_noise(size, device):
    """ Generates a noise tensor. """
    
    return torch.randn(size, device=device)

