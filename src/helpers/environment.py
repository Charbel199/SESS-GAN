import numpy as np
import torch
from utils import load_np
import glob
import os


def load_environments(environments_path, tokens, data_format) -> torch.Tensor:
    raw_environments = glob.glob(os.path.join(environments_path, f"*.{data_format}"))
    environments = []
    for raw_env in raw_environments:
        data = load_np(raw_env, data_format)
        data = np.nan_to_num(data, nan=0)
        one_hot_environment = tokens_to_one_hot_environment(data, tokens)
        environments.append(one_hot_environment)
    environments = torch.stack(environments)

    return environments


def tokens_to_one_hot_environment(level, tokens):
    """ Converts an environment of token to a full one-hot token tensor. """
    environment = torch.zeros((len(tokens), len(level), len(level[-1])))
    for i in range(len(level)):
        for j in range(len(level[-1])):
            token = str(int(level[i][j]))
            if token in tokens and token != "\n":
                environment[tokens.index(token), i, j] = 1

    return environment


def one_hot_environment_to_tokens(level):
    """ Converts a full token level tensor to an ascii level. """
    environment = torch.argmax(level, dim=0)
    return environment
