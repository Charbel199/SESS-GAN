import torch
from utils import load_np
import glob
import os
from typing import List
import torch.nn.functional as F


def load_environments(environments_path, tokens, data_format) -> List:
    raw_environments = glob.glob(os.path.join(environments_path, f"*.{data_format}"))
    environments = []
    for raw_env in raw_environments:
        data = load_np(raw_env, data_format)
        one_hot_environment = tokens_to_one_hot_environment(data, tokens)
        environments.append(one_hot_environment)
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
