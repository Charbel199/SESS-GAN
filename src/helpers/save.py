import torch
import os
from logger.log import LoggerService

logger = LoggerService.get_instance()


def torch_save(data, path: str, file_name: str) -> None:
    if not os.path.exists(path):
        # Create a new directory because it does not exist
        os.makedirs(path)
        logger.info(f'Created directory {path}')

    torch.save(data, os.path.join(path, file_name))

