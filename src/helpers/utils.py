import numpy as np


def load_np(path, file_type) -> np.ndarray:
    if file_type == 'npy':
        return np.load(f'{path}')
    if file_type == 'txt':
        return np.genfromtxt(f'{path}')
