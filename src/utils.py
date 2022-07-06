import numpy as np


def load_np(name) -> np.ndarray:
    return np.load(f'{name}.npy')  # load
