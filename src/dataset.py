import numpy as np
from torch.utils.data import Dataset
import glob, os
import torch
from torch.utils.data import DataLoader
from utils import load_np


class EnvironmentDataset(Dataset):
    def __init__(self, data_dir, data_format='txt', transform=None):
        self.data_dir = data_dir
        self.data_format = data_format
        self.transform = transform
        self.items = glob.glob(os.path.join(data_dir, f"*.{data_format}"))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        data = load_np(self.items[index], self.data_format)

        if self.transform is not None:
            augmentations = self.transform(image=data)
            data = augmentations["image"]

        data = torch.from_numpy(data).int()  # Parsing tokens

        return data


def get_loaders(
        train_dir,
        val_dir,
        batch_size,
        train_transform,
        val_transform,
        data_format='txt',
        num_workers=4,
        pin_memory=True,
):
    train_ds = EnvironmentDataset(
        data_dir=train_dir,
        transform=train_transform,
        data_format=data_format
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = EnvironmentDataset(
        data_dir=val_dir,
        transform=val_transform,
        data_format=data_format
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader
