import numpy as np
import json


def load_np(path, file_type) -> np.ndarray:
    if file_type == 'npy':
        return np.load(f'{path}')
    if file_type == 'txt':
        return np.genfromtxt(f'{path}')


def object_variables_to_json(obj, output_path):
    config_params = vars(obj)
    config_params = json.dumps(config_params, default=str)
    with open(output_path, 'w') as f:
        f.write(config_params)
