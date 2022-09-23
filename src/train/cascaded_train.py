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

logger = LoggerService.get_instance()


class CascadedModelTrainer(ModelTrainer):

    def train_model(self, config: ModelConfig):
        if config.load:
            pass
        else:
            generators = []
            noise_maps = []
            noise_amplitudes = []

            # Set token group
            if config.environment == Environment.ROBOT_NAVIGATION:
                config.token_list = robot_navigation_token_list

            # Get real environments
            real = load_environments(config.train_path, config.token_list, 'txt')  # TODO: Import environment

            # Downsample original environments
            scaled_list = robot_navigation_downsampling(config.scales, real, config.token_list)

            # Generate a list of all real input environments (Scaled & Original)
            reals = [*scaled_list, real]

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
                torch_save(generators, config.output_path, 'noise_maps.pth')
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

        padsize = int(config.number_of_layers)  # As kernel size is always 3 currently, padsize goes up by one per layer

        if config.pad_type == PaddingType.ZERO:
            pad_noise = nn.ZeroPad2d(padsize)
            pad_image = nn.ZeroPad2d(padsize)
        elif config.pad_type == PaddingType.REFLECTION:
            pad_noise = nn.ReflectionPad2d(padsize)
            pad_image = nn.ReflectionPad2d(padsize)

        # setup optimizer
        optimizerD = optim.Adam(D.parameters(), lr=config.learning_rate_discriminator, betas=(config.beta1, 0.999))
        optimizerG = optim.Adam(G.parameters(), lr=config.learning_rate_generator, betas=(config.beta1, 0.999))
        schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD, milestones=[1600, 2500],
                                                          gamma=config.gamma)
        schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG, milestones=[1600, 2500],
                                                          gamma=config.gamma)

        if current_scale == 0:  # Generate new noise
            z_opt = generate_spatial_noise([1, len(config.token_list), noise_size_x, noise_size_y],
                                           device=config.device)
            z_opt = pad_noise(z_opt)
        else:  # Add noise to previous output
            z_opt = torch.zeros([1, len(config.token_list), noise_size_x, noise_size_y]).to(config.device)
            z_opt = pad_noise(z_opt)

        logger.info(f"Training at scale {current_scale}")
        for epoch in tqdm(range(config.epoch)):
            step = current_scale * config.epoch + epoch
            noise_ = generate_spatial_noise([1, len(config.token_list), noise_size_x, noise_size_y],
                                            device=config.device)

            noise_ = pad_noise(noise_)

            ############################
            # (1) Update D network: maximize D(x) + D(G(z))
            ###########################
            for j in range(config.discriminator_steps):
                # train with real
                D.zero_grad()

                output = D(current_scale_real).to(config.device)

                errD_real = -output.mean()
                errD_real.backward(retain_graph=True)

                # train with fake
                if (j == 0) & (epoch == 0):
                    if current_scale == 0:  # If we are in the lowest scale, noise is generated from scratch
                        prev = torch.zeros(1, len(config.token_list), noise_size_x, noise_size_y).to(config.device)
                        input_from_prev_scale = prev
                        prev = pad_image(prev)
                        z_prev = torch.zeros(1, len(config.token_list), noise_size_x, noise_size_y).to(config.device)
                        z_prev = pad_noise(z_prev)
                        config.noise_amp = 1
                    else:  # First step in NOT the lowest scale
                        # We need to adapt our inputs from the previous scale and add noise to it
                        prev = draw_concat(generators, noise_maps, reals, noise_amplitudes, input_from_prev_scale,
                                           "rand", pad_noise, pad_image, config)

                        prev = interpolate(prev, current_scale_real.shape[-2:], mode="bilinear", align_corners=False)
                        prev = pad_image(prev)

                        z_prev = draw_concat(generators, noise_maps, reals, noise_amplitudes, input_from_prev_scale,
                                             "rec", pad_noise, pad_image, config)

                        z_prev = interpolate(z_prev, current_scale_real.shape[-2:], mode="bilinear",
                                             align_corners=False)
                        config.noise_amp = update_noise_amplitude(z_prev, current_scale_real, config)
                        z_prev = pad_image(z_prev)
                else:  # Any other step
                    prev = draw_concat(generators, noise_maps, reals, noise_amplitudes, input_from_prev_scale,
                                       "rand", pad_noise, pad_image, config)

                    prev = interpolate(prev, current_scale_real.shape[-2:], mode="bilinear", align_corners=False)
                    prev = pad_image(prev)

                # After creating our correct noise input, we feed it to the generator:
                noise = config.noise_amp * noise_ + prev
                fake = G(noise.detach(), prev, temperature=1)

                # Then run the result through the discriminator
                output = D(fake.detach())
                errD_fake = output.mean()

                # Backpropagation
                errD_fake.backward(retain_graph=False)

                # Gradient Penalty
                gradient_penalty = calc_gradient_penalty(D, current_scale_real, fake, config.lambda_grad, config.device)
                gradient_penalty.backward(retain_graph=False)

                # Logging:
                # if step % 10 == 0:
                #     wandb.log({f"D(G(z))@{current_scale}": errD_fake.item(),
                #                f"D(x)@{current_scale}": -errD_real.item(),
                #                f"gradient_penalty@{current_scale}": gradient_penalty.item()
                #                },
                #               step=step, sync=False)
                optimizerD.step()

            ############################
            # (2) Update G network: maximize D(G(z))
            ###########################

            for j in range(config.generator_steps):
                G.zero_grad()
                fake = G(noise.detach(), prev.detach(), temperature=1)
                output = D(fake)

                errG = -output.mean()
                errG.backward(retain_graph=False)
                if config.alpha != 0:  # i. e. we are trying to find an exact recreation of our input in the lat space
                    Z_opt = config.noise_amp * z_opt + z_prev
                    G_rec = G(Z_opt.detach(), z_prev, temperature=1)
                    rec_loss = config.alpha * F.mse_loss(G_rec, current_scale_real)
                    rec_loss.backward(retain_graph=False)  # TODO: Check for unexpected argument retain_graph=True
                    rec_loss = rec_loss.detach()
                else:  # We are not trying to find an exact recreation
                    rec_loss = torch.zeros([])
                    Z_opt = z_opt

                optimizerG.step()

            # More Logging:
            # if step % 10 == 0:
            #     wandb.log({f"noise_amplitude@{current_scale}": opt.noise_amp,
            #                f"rec_loss@{current_scale}": rec_loss.item()},
            #               step=step, sync=False, commit=True)

            # Rendering and logging images of levels
            # if epoch % 500 == 0 or epoch == (opt.niter - 1):
            #     if opt.token_insert >= 0 and opt.nc_current == len(token_group):
            #         token_list = [list(group.keys())[0] for group in token_group]
            #     else:
            #         token_list = opt.token_list
            #
            #     img = opt.ImgGen.render(one_hot_to_ascii_level(fake.detach(), token_list))
            #     img2 = opt.ImgGen.render(one_hot_to_ascii_level(
            #         G(Z_opt.detach(), z_prev, temperature=1 if current_scale != opt.token_insert else 1).detach(),
            #         token_list))
            #     real_scaled = one_hot_to_ascii_level(current_scale_real.detach(), token_list)
            #     img3 = opt.ImgGen.render(real_scaled)
            #     wandb.log({f"G(z)@{current_scale}": wandb.Image(img),
            #                f"G(z_opt)@{current_scale}": wandb.Image(img2),
            #                f"real@{current_scale}": wandb.Image(img3)},
            #               sync=False, commit=False)
            #
            #     real_scaled_path = os.path.join(wandb.run.dir, f"real@{current_scale}.txt")
            #     with open(real_scaled_path, "w") as f:
            #         f.writelines(real_scaled)
            #     wandb.save(real_scaled_path)

            # Learning Rate scheduler step
            schedulerD.step()
            schedulerG.step()

        # Save networks

        torch_save(G.state_dict(), config.current_output_path, 'G.pth')
        torch_save(D.state_dict(), config.current_output_path, 'D.pth')
        torch_save(z_opt, config.current_output_path, 'z_opt.pth')

        # wandb.save(opt.outf)
        return z_opt, input_from_prev_scale, G
