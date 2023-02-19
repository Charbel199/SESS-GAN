import torch
from conf import ModelConfig
from logger.log import LoggerService
import os
from train.abstract_train import ModelTrainer
from environments.robot_navigation.downsampling import robot_navigation_downsampling
from environments.robot_navigation.tokens import TOKEN_LIST as robot_navigation_token_list
from model.singan.models_helper import initialize_new_models, reset_gradients
from typing import List
import torch.nn as nn
from enums.environment import Environment
from enums.padding_type import PaddingType
import torch.optim as optim
from tqdm import tqdm
from helpers.generate_noise import generate_spatial_noise
from model.singan.gradient_penalty import calc_gradient_penalty
import torch.nn.functional as F
from torch.nn.functional import interpolate
from helpers.draw_concat import draw_concat
from helpers.noise import update_noise_amplitude
from helpers.environment import load_environments, one_hot_environment_to_tokens
from helpers.save import torch_save
from helpers.utils import object_variables_to_json

logger = LoggerService.get_instance()


class CascadedModelTrainer(ModelTrainer):

    def train_model(self, config: ModelConfig):
        # Create output directory
        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)

        # Save config
        object_variables_to_json(config, os.path.join(config.output_path))

        generators = []
        noise_maps = []
        noise_amplitudes = []

        # Set token group
        if config.environment == Environment.ROBOT_NAVIGATION:
            config.token_list = robot_navigation_token_list

        # Get real environments
        real = load_environments(config.train_path, config.token_list, 'txt')

        # Downsample original environments
        scaled_list = robot_navigation_downsampling(config.scales, real, config.token_list)

        # Generate a list of all real input environments (Scaled & Original)
        reals = [*scaled_list, real]

        # import matplotlib.pyplot as plt
        # from helpers.environment import one_hot_environment_to_tokens
        # for s in reals:
        #     plt.imshow(one_hot_environment_to_tokens(s[0]))
        #     plt.show()

        # Generate tensor of 0 (since we are starting with the first scale and there is no previous scale inputs)
        # with the size of the first scale
        input_from_prev_scale = torch.zeros_like(reals[0])

        # Log the original input level as an image
        # img = opt.ImgGen.render(one_hot_to_ascii_level(real, opt.token_list))
        # wandb.log({"real": wandb.Image(img)}, commit=False)
        # os.makedirs("%s/state_dicts" % (opt.out_), exist_ok=True)

        # Training Loop
        for current_scale in range(0, config.total_num_of_scales):
            # Create directory for current scale
            config.current_output_path = os.path.join(config.output_path, f"scale_{current_scale}")
            try:
                os.makedirs(config.current_output_path)
            except OSError:
                pass

            # Initialize models
            D, G = initialize_new_models(config)

            # Train the current scale
            z_opt, input_from_prev_scale, G = self._train_scale(D, G, reals, generators, noise_maps,
                                                                input_from_prev_scale, noise_amplitudes,
                                                                current_scale, config)

            # Reset grads and save current scale
            G = reset_gradients(G, False)
            G.eval()
            D = reset_gradients(D, False)
            D.eval()

            generators.append(G)
            noise_maps.append(z_opt)
            noise_amplitudes.append(config.noise_amp)

            # Saving variables
            torch_save(noise_maps, config.output_path, 'noise_maps.pth')
            torch_save(generators, config.output_path, 'generators.pth')
            torch_save(reals, config.output_path, 'reals.pth')
            torch_save(noise_amplitudes, config.output_path, 'noise_amplitudes.pth')
            torch_save(config.number_of_layers, config.output_path, 'num_layer.pth')
            torch_save(config.token_list, config.output_path, 'token_list.pth')
            torch_save(G.state_dict(), os.path.join(config.output_path, 'state_dicts'), f"G_{current_scale}.pth")

            del D, G

        return generators, noise_maps, reals, noise_amplitudes

    def _train_scale(self, D, G, reals: List, generators: List, noise_maps: List,
                     input_from_prev_scale: torch.Tensor, noise_amplitudes: List, current_scale: int,
                     config: ModelConfig):
        """ Train one scale. D and G are the current discriminator and generator, reals are the scaled versions of the
           original level, generators and noise_maps contain information from previous scales and will receive information in
           this scale, input_from_previous_scale holds the noise map and images from the previous scale, noise_amplitudes hold
           the amplitudes for the noise in all the scales. opt is a namespace that holds all necessary parameters. """

        current_scale_real = reals[current_scale].to(config.device)

        # Set token list
        if config.environment == Environment.ROBOT_NAVIGATION:
            config.token_list = robot_navigation_token_list

        noise_size_x = current_scale_real.shape[2]  # Noise size x
        noise_size_y = current_scale_real.shape[3]  # Noise size y

        pad_size = int(
            config.number_of_layers * ((config.kernel_size - 1) / 2))  # Padding to have same input and output sizes

        if config.pad_type == PaddingType.ZERO:
            pad_tensor = nn.ZeroPad2d(pad_size)
        elif config.pad_type == PaddingType.REFLECTION:
            pad_tensor = nn.ReflectionPad2d(pad_size)
        else:
            # Default, no padding
            pad_tensor = nn.ZeroPad2d(0)

        # setup optimizer
        optimizer_discriminator = optim.Adam(D.parameters(), lr=config.learning_rate_discriminator,
                                             betas=(config.beta1, 0.999))
        optimizer_generator = optim.Adam(G.parameters(), lr=config.learning_rate_generator, betas=(config.beta1, 0.999))
        scheduler_discriminator = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer_discriminator,
                                                                       milestones=[1600, 2500],
                                                                       gamma=config.gamma)
        scheduler_generator = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer_generator,
                                                                   milestones=[1600, 2500],
                                                                   gamma=config.gamma)

        if current_scale == 0:  # Generate new noise
            noise_map = generate_spatial_noise([1, len(config.token_list), noise_size_x, noise_size_y],
                                               device=config.device)
            noise_map = pad_tensor(noise_map)
        else:  # Add noise to previous output
            noise_map = torch.zeros([1, len(config.token_list), noise_size_x, noise_size_y]).to(config.device)
            noise_map = pad_tensor(noise_map)

        logger.info(f"Training at scale {current_scale}")
        for current_epoch in tqdm(range(config.epoch)):
            step = current_scale * config.epoch + current_epoch
            noise_ = generate_spatial_noise([1, len(config.token_list), noise_size_x, noise_size_y],
                                            device=config.device)

            noise_ = pad_tensor(noise_)

            ############################
            # (1) Update D network: maximize D(x) + D(G(z))
            ###########################
            for j in range(config.discriminator_steps):
                # train with real
                D.zero_grad()

                output = D(current_scale_real).to(config.device)

                error_discriminator_real = -output.mean()
                error_discriminator_real.backward(retain_graph=True)

                # train with fake
                if (j == 0) & (current_epoch == 0):
                    if current_scale == 0:  # If we are in the lowest scale, noise is generated from scratch
                        previous_input = torch.zeros(1, len(config.token_list), noise_size_x, noise_size_y).to(
                            config.device)
                        input_from_prev_scale = previous_input
                        previous_input = pad_tensor(previous_input)
                        previous_noise_map = torch.zeros(1, len(config.token_list), noise_size_x, noise_size_y).to(
                            config.device)
                        previous_noise_map = pad_tensor(previous_noise_map)
                        config.noise_amp = 1
                    else:  # First step in NOT the lowest scale
                        # We need to adapt our inputs from the previous scale and add noise to it
                        previous_input = draw_concat(generators, noise_maps, reals, noise_amplitudes,
                                                     input_from_prev_scale,
                                                     "rand", pad_tensor, pad_tensor, config)

                        previous_input = interpolate(previous_input, current_scale_real.shape[-2:], mode="bilinear",
                                                     align_corners=False)
                        previous_input = pad_tensor(previous_input)

                        previous_noise_map = draw_concat(generators, noise_maps, reals, noise_amplitudes,
                                                         input_from_prev_scale,
                                                         "rec", pad_tensor, pad_tensor, config)

                        previous_noise_map = interpolate(previous_noise_map, current_scale_real.shape[-2:],
                                                         mode="bilinear",
                                                         align_corners=False)
                        config.noise_amp = update_noise_amplitude(previous_noise_map, current_scale_real, config)
                        previous_noise_map = pad_tensor(previous_noise_map)
                else:  # Any other step
                    previous_input = draw_concat(generators, noise_maps, reals, noise_amplitudes, input_from_prev_scale,
                                                 "rand", pad_tensor, pad_tensor, config)

                    previous_input = interpolate(previous_input, current_scale_real.shape[-2:], mode="bilinear",
                                                 align_corners=False)
                    previous_input = pad_tensor(previous_input)

                # After creating our correct noise input, we feed it to the generator:
                noise = config.noise_amp * noise_ + previous_input
                fake = G(noise.detach(), previous_input, temperature=1)

                # Then run the result through the discriminator
                output = D(fake.detach())
                error_discriminator_fake = output.mean()

                # Backpropagation
                error_discriminator_fake.backward(retain_graph=False)

                # Gradient Penalty
                gradient_penalty = calc_gradient_penalty(D, current_scale_real, fake, config.lambda_grad, config.device)
                gradient_penalty.backward(retain_graph=False)

                # Logging:
                if step % 200 == 0:
                    logger.info(f"D(G(z))@scale_{current_scale}: {error_discriminator_fake.item()} \n"
                                f"D(x)@scale_{current_scale}: {-error_discriminator_real.item()} \n"
                                f"gradient_penalty@scale_{current_scale}: {gradient_penalty.item()}"
                                )
                optimizer_discriminator.step()

            ############################
            # (2) Update G network: maximize D(G(z))
            ###########################

            for j in range(config.generator_steps):
                G.zero_grad()
                fake = G(noise.detach(), previous_input.detach(), temperature=1)
                output = D(fake)

                errG = -output.mean()
                errG.backward(retain_graph=False)
                if config.alpha != 0:  # Example: we are trying to find an exact recreation of our input in the lat space
                    Z_opt = config.noise_amp * noise_map + previous_noise_map
                    G_rec = G(Z_opt.detach(), previous_noise_map, temperature=1)
                    rec_loss = config.alpha * F.mse_loss(G_rec, current_scale_real)
                    rec_loss.backward(retain_graph=False)  # TODO: Check for unexpected argument retain_graph=True
                    rec_loss = rec_loss.detach()
                else:  # We are not trying to find an exact recreation
                    rec_loss = torch.zeros([])
                    Z_opt = noise_map

                optimizer_generator.step()

            # More Logging:
            if step % 200 == 0:
                logger.info(f"noise_amplitude@sacle_{current_scale}: {config.noise_amp}"
                            f"rec_loss@scale_{current_scale}: {rec_loss.item()}")

            # Rendering and logging images of levels
            if current_epoch % 500 == 0 or current_epoch == (config.epoch - 1):
                dir2save = os.path.join(config.output_path, 'training_samples')
                # Make directories
                try:
                    os.makedirs(dir2save, exist_ok=True)
                    os.makedirs(os.path.join(dir2save, 'img'), exist_ok=True)
                    os.makedirs(os.path.join(dir2save, 'txt'), exist_ok=True)
                except OSError:
                    pass
                fake = G(noise.detach(), previous_input.detach(), temperature=1)
                # Convert to ascii level
                level = one_hot_environment_to_tokens(fake[0].detach().cpu())

                # Save level txt
                torch_save(level, os.path.join(dir2save, 'txt'), f"scale{current_scale}_step{step}.txt",
                           file_type='txt')
                # Save level image
                torch_save(level, os.path.join(dir2save, 'img'), f"scale{current_scale}_step{step}.png",
                           file_type='image')

            # Learning Rate scheduler step
            scheduler_discriminator.step()
            scheduler_generator.step()

        # Save networks

        torch_save(G.state_dict(), config.current_output_path, 'G.pth')
        torch_save(D.state_dict(), config.current_output_path, 'D.pth')
        torch_save(noise_map, config.current_output_path, 'z_opt.pth')

        # wandb.save(opt.outf)
        return noise_map, input_from_prev_scale, G
