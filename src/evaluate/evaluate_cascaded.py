import os
from conf import ModelConfig
from model.singan.models_helper import load_trained_components
from evaluate.generate_samples import generate_samples
from helpers.environment import load_environments


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
                         gen_start_scale=config.starting_scale, num_samples=1000, render_images=False,
                         save_tensors=False, save_dir=save_directory)
