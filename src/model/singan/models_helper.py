import torch
from model.singan.discriminator import Discriminator
from model.singan.generator import Generator
from conf import ModelConfig
from model.weights import conv_weights_init
import os
from logger.log import LoggerService

logger = LoggerService.get_instance()


def initialize_new_models(opt: ModelConfig):
    """ Initialize Generator and Discriminator. """
    # generator initialization:
    G = Generator(opt).to(opt.device)
    conv_weights_init(G)
    # G.apply(weights_init)
    # if opt.netG != "":
    #     G.load_state_dict(torch.load(opt.netG))
    logger.info(f"Initialize new generator \n {G}")

    # discriminator initialization:
    D = Discriminator(opt).to(opt.device)
    conv_weights_init(D)
    # D.apply(weights_init)
    # if opt.netD != "":
    #     D.load_state_dict(torch.load(opt.netD))
    logger.info(f"Initialize new discriminator \n {D}")

    return D, G


def reset_gradients(model, require_grad):
    for p in model.parameters():
        p.requires_grad_(require_grad)
    return model


def load_trained_components(config: ModelConfig):
    if os.path.exists(config.output_path):
        logger.info(f'Loading components from {config.output_path}')
        real_environments = torch.load(os.path.join(config.output_path, 'reals.pth'))
        generators = torch.load(os.path.join(config.output_path, 'generators.pth'))
        noise_maps = torch.load(os.path.join(config.output_path, 'noise_maps.pth'))
        noise_amplitudes = torch.load(os.path.join(config.output_path, 'noise_amplitudes.pth'))
        logger.info(f'Done loading components from {config.output_path}')
        return generators, noise_maps, real_environments, noise_amplitudes
    else:
        logger.info(f'Tried loading components from {config.output_path} but directory does not exist')
        return
