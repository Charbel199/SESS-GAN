from conf import ModelConfig
from logger.log import LoggerService
import os
import torch.nn as nn
from tqdm import tqdm
from helpers.generate_noise import generate_spatial_noise
from torch.nn.functional import interpolate
from enums.padding_type import PaddingType
from enums.environment import Environment
from environments.robot_navigation.tokens import TOKEN_LIST as robot_navigation_token_list
from helpers.environment import one_hot_environment_to_tokens
import torch
from helpers.save import torch_save

logger = LoggerService.get_instance()


def generate_samples(generators, noise_maps, reals, noise_amplitudes, config: ModelConfig, input_image=None,
                     scale_v=1.0, scale_h=1.0,
                     current_scale=0, gen_start_scale=0, num_samples=50, render_images=True, save_tensors=False,
                     save_dir="random_samples"):
    # Holds images generated in current scale
    current_images = []

    # Set token list
    if config.environment == Environment.ROBOT_NAVIGATION:
        config.token_list = robot_navigation_token_list

    # Main sampling loop
    for G, Z_opt, noise_amp in zip(generators, noise_maps, noise_amplitudes):

        if current_scale >= len(generators):
            break  # if we do not start at current_scale=0 we need this

        logger.info(f"Generating samples at scale {current_scale}")

        # Padding (should be chosen according to what was trained with)
        padsize = int(
            1 * config.number_of_layers)  # As kernel size is always 3 currently, padsize goes up by one per layer
        if config.pad_type == PaddingType.ZERO:
            pad_tensor = nn.ZeroPad2d(padsize)
        elif config.pad_type == PaddingType.REFLECTION:
            pad_tensor = nn.ReflectionPad2d(padsize)

        # Calculate shapes to generate
        if 0 < gen_start_scale <= current_scale:  # Special case! Can have a wildly different shape through in_s
            scale_v = input_image.shape[-2] / (noise_maps[gen_start_scale - 1].shape[-2] - padsize * 2)
            scale_h = input_image.shape[-1] / (noise_maps[gen_start_scale - 1].shape[-1] - padsize * 2)
            noise_size_x = (Z_opt.shape[-2] - padsize * 2) * scale_v
            noise_size_y = (Z_opt.shape[-1] - padsize * 2) * scale_h
        else:
            noise_size_x = (Z_opt.shape[-2] - padsize * 2) * scale_v
            noise_size_y = (Z_opt.shape[-1] - padsize * 2) * scale_h

        # Save list of images of previous scale and clear current images
        images_prev = current_images
        current_images = []

        # If input_image is none or filled with zeros reshape to correct size with channels
        if input_image is None:
            input_image = torch.zeros(reals[0].shape[0], len(config.token_list), *reals[0].shape[2:]).to(config.device)
        elif input_image.sum() == 0:
            input_image = torch.zeros(1, len(config.token_list), *input_image.shape[-2:]).to(config.device)

        # Generate num_samples samples in current scale
        for n in tqdm(range(0, num_samples, 1)):

            # Get noise image
            z_curr = generate_spatial_noise(
                [1, len(config.token_list), int(round(noise_size_x)), int(round(noise_size_y))], device=config.device)
            z_curr = pad_tensor(z_curr)

            # Set up previous image I_prev
            if (not images_prev) or current_scale == 0:  # if there is no "previous" image
                previous_image = input_image
            else:
                previous_image = images_prev[n]

            previous_image = interpolate(previous_image, [int(round(noise_size_x)), int(round(noise_size_y))],
                                         mode='bilinear', align_corners=False)
            previous_image = pad_tensor(previous_image)

            # We take the optimized noise map Z_opt as an input if we start generating on later scales
            if current_scale < gen_start_scale:
                z_curr = Z_opt

            # Generate
            z_in = noise_amp * z_curr + previous_image
            current_image = G(z_in.detach(), previous_image, temperature=1)

            # Save all scales
            # if True:

            # Save scale 0 and last scale
            # if current_scale == 0 or current_scale == len(reals) - 1:

            # Save only last scale
            if current_scale == len(reals) - 1:
                dir2save = os.path.join(config.output_path, save_dir)

                # Make directories
                try:
                    os.makedirs(dir2save, exist_ok=True)
                    if render_images:
                        os.makedirs(os.path.join(dir2save, 'img'), exist_ok=True)
                    if save_tensors:
                        os.makedirs(os.path.join(dir2save, 'torch'), exist_ok=True)
                    os.makedirs(os.path.join(dir2save, 'txt'), exist_ok=True)
                except OSError:
                    pass

                # Convert to ascii level
                level = one_hot_environment_to_tokens(current_image[0].detach().cpu())

                # Save level txt
                torch_save(level, os.path.join(dir2save, 'txt'), f"{n}_scale{current_scale}.txt", file_type='txt')
                # Save level image
                torch_save(level, os.path.join(dir2save, 'img'), f"{n}_scale{current_scale}.png",
                           file_type='image')

            # Append current image
            current_images.append(current_image)

        # Go to next scale
        current_scale += 1

    return current_image.detach()  # return last generated image
