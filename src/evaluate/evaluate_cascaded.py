import os
from conf import ModelConfig
from model.singan.models_helper import load_trained_components
from evaluate.generate_samples import generate_samples
from helpers.environment import load_environments
from logger.log import LoggerService
import collections
logger = LoggerService.get_instance()

def evaluate_cascaded(config: ModelConfig):
    # Get real environments
    reals = load_environments(config.train_path, config.token_list, 'txt')

    for real_m in reals:
        # Load components
        generators_m, noise_maps_m, reals_m, noise_amplitudes_m = load_trained_components(config)

        # Set in_s and scales
        if config.starting_scale == 0:  # starting in lowest scale
            input_image = None
            scale_v = 1.0
            # scale_h = 200 / real_m.shape[-1]  # normalize all levels to length 16x200
            scale_h = 1.0
        else:  # if opt.starting_scale > 0
            # Only works with default level size if no in_s is provided (should not be reached)
            input_image = reals_m[config.starting_scale]
            scale_v = 1.0
            scale_h = 1.0

        # Define directory
        save_directory = f"random_samples"

        # Generate samples
        generate_samples(generators_m, noise_maps_m, reals_m, noise_amplitudes_m, config, input_image=input_image,
                         scale_v=scale_v, scale_h=scale_h, current_scale=config.starting_scale,
                         gen_start_scale=config.starting_scale, num_samples=50, render_images=False,
                         save_tensors=False, save_dir=save_directory)

from functools import partial
import math
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
from helpers.environment import load_environments
from environments.robot_navigation.tokens import TOKEN_LIST as robot_navigation_token_list
def pattern_key(level_slice):
    """
    Computes a hashable key from a level slice.
    """
    key = ""
    for line in level_slice:
        for token in line:
            key += str(token)
    return key
def get_pattern_counts(level, pattern_size):
    """
    Collects counts from all patterns in the level of the given size.
    """
    pattern_counts = collections.defaultdict(int)
    for up in range(level.shape[0] - pattern_size + 1):
        for left in range(level.shape[1] - pattern_size + 1):
            down = up + pattern_size
            right = left + pattern_size
            level_slice = level[up:down, left:right]
            pattern_counts[pattern_key(level_slice)] += 1
    return pattern_counts
def compute_pattern_counts(dataset, pattern_size):
    """
    Compute pattern counts in parallel from a given dataset.
    """
    levels = [level.argmax(dim=0).numpy() for level in dataset]
    with mp.Pool() as pool:
        counts_per_level = pool.map(
            partial(get_pattern_counts, pattern_size=pattern_size), levels,
        )
    pattern_counts = collections.defaultdict(int)
    for counts in counts_per_level:
        for pattern, count in counts.items():
            pattern_counts[pattern] += count
    return pattern_counts

def compute_prob(pattern_count, num_patterns, epsilon=1e-7):
    """
    Compute probability of a pattern.
    """
    return (pattern_count + epsilon) / ((num_patterns + epsilon) * (1 + epsilon))

from conf import parse_args
def compute_kl_divergence():
    hparams =  parse_args()
    hparams.slice_width = 2
    hparams.pattern_sizes = [3, 5, 7]
    hparams.weight = 0.8

    test_dataset = load_environments('assets/results5/random_samples/txt', robot_navigation_token_list, 'txt')
    dataset = load_environments('assets/data/train3', robot_navigation_token_list, 'txt')
    kl_divergences = []
    for pattern_size in hparams.pattern_sizes:
        logger.info("Computing original pattern counts...")
        pattern_counts = compute_pattern_counts(dataset, pattern_size)
        logger.info("Computing test pattern counts...")
        test_pattern_counts = compute_pattern_counts(
            test_dataset, pattern_size)

        num_patterns = sum(pattern_counts.values())
        num_test_patterns = sum(test_pattern_counts.values())
        logger.info(
            "Found {} patterns and {} test patterns", num_patterns, num_test_patterns
        )

        kl_divergence = 0
        for pattern, count in tqdm(pattern_counts.items()):
            prob_p = compute_prob(count, num_patterns)
            prob_q = compute_prob(
                test_pattern_counts[pattern], num_test_patterns)
            kl_divergence += hparams.weight * prob_p * math.log(prob_p / prob_q) + (
                1 - hparams.weight
            ) * prob_q * math.log(prob_q / prob_p)

        kl_divergences.append(kl_divergence)
        logger.info(
            "KL-Divergence @ {}x{}: {}",
            pattern_size,
            pattern_size,
            round(kl_divergence, 2),
        )
    mean_kl_divergence = np.mean(kl_divergences)
    var_kl_divergence = np.std(kl_divergences)
    logger.info(f"Average KL-Divergence: {round(mean_kl_divergence, 2)}")
    return mean_kl_divergence, var_kl_divergence

if __name__ == "__main__":
    x = compute_kl_divergence()
