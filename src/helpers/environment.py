import torch
from utils import load_np
import glob
import os
from typing import List


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
    # TODO: Add logic
    return environment
