import numpy as np


def load_np(path, extension) -> np.ndarray:
    if extension == 'npy':
        return np.load(f'{path}.{extension}')
    if extension == 'txt':
        return np.genfromtxt(f'{path}.{extension}')
