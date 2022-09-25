import torch
import os
from logger.log import LoggerService
import numpy as np
import matplotlib.pyplot as plt

logger = LoggerService.get_instance()


def torch_save(data, path: str, file_name: str, file_type='pth') -> None:
    if not os.path.exists(path):
        # Create a new directory because it does not exist
        os.makedirs(path)
        logger.info(f'Created directory {path}')

    if file_type == 'pth':
        torch.save(data, os.path.join(path, file_name))
    if file_type == 'txt':
        np.savetxt(os.path.join(path, file_name), data.numpy().astype(int), fmt='%s',
                   delimiter=" ")
    if file_type == 'image':
        plt.imshow(data)
        plt.savefig(os.path.join(path, file_name))
