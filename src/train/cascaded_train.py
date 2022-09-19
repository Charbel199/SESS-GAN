import torch
from conf import ModelConfig
from logger.log import LoggerService
import os
from train.abstract_train import ModelTrainer
from environments.robot_navigation.downsampling import robot_navigation_downsampling
from environments.robot_navigation.tokens import TOKEN_LIST as robot_navigation_token_list
from model.singan.models_helper import initialize_new_models, reset_gradients

logger = LoggerService.get_instance()


class CascadedModelTrainer(ModelTrainer):

    def train_model(self, config: ModelConfig):
        if config.load:
            pass
        else:
            generators = []
            noise_maps = []
            noise_amplitudes = []

            # Get real environments
            real = 'image' # TODO: Import environment

            # Set token list
            if config.environment == 'robot-navigation':
                config.token_list = robot_navigation_token_list

            scaled_list = robot_navigation_downsampling(config.scales, real, config.token_list)

            # Generate a list of all real input environments (Scaled & Original)
            reals = [*scaled_list, real]

            input_from_prev_scale = torch.zeros_like(reals[0])

            # Log the original input level as an image
            # img = opt.ImgGen.render(one_hot_to_ascii_level(real, opt.token_list))
            # wandb.log({"real": wandb.Image(img)}, commit=False)
            # os.makedirs("%s/state_dicts" % (opt.out_), exist_ok=True)

            # Training Loop
            for current_scale in range(0, config.total_num_of_scales):
                # Create directory for current scale
                current_output_path = os.path.join(config.output_path, f"scale_{current_scale}")
                try:
                    os.makedirs(current_output_path)
                except OSError:
                    pass

                # Initialize models
                D, G = initialize_new_models(config)

                # Actually train the current scale
                z_opt, input_from_prev_scale, G = train_single_scale(D, G, reals, generators, noise_maps,
                                                                     input_from_prev_scale, noise_amplitudes, opt)

                # Reset grads and save current scale
                G = reset_gradients(G, False)
                G.eval()
                D = reset_gradients(D, False)
                D.eval()

                generators.append(G)
                noise_maps.append(z_opt)
                noise_amplitudes.append(config.noise_amp)

                # Saving variables
                torch.save(noise_maps, os.path.join(config.output_path, 'noise_maps.pth'))
                torch.save(generators, os.path.join(config.output_path, 'noise_maps.pth'))
                torch.save(reals, os.path.join(config.output_path, 'reals.pth'))
                torch.save(noise_amplitudes, os.path.join(config.output_path, 'noise_amplitudes.pth'))
                torch.save(config.number_of_layers, os.path.join(config.output_path, 'num_layer.pth'))
                torch.save(config.token_list, os.path.join(config.output_path, 'token_list.pth'))
                torch.save(G.state_dict(), os.path.join(config.output_path, 'state_dicts', f"G_{current_scale}.pth"))

                del D, G

            return generators, noise_maps, reals, noise_amplitudes

