import torch
from model.singan.discriminator import Discriminator
from model.singan.generator import Generator
from conf import ModelConfig
from model.weights import conv_weights_init


def initialize_new_models(opt: ModelConfig):
    """ Initialize Generator and Discriminator. """
    # generator initialization:
    G = Generator(opt).to(opt.device)
    conv_weights_init(G)
    # G.apply(weights_init)
    # if opt.netG != "":
    #     G.load_state_dict(torch.load(opt.netG))
    print(G)

    # discriminator initialization:
    D = Discriminator(opt).to(opt.device)
    conv_weights_init(D)
    # D.apply(weights_init)
    # if opt.netD != "":
    #     D.load_state_dict(torch.load(opt.netD))
    print(D)

    return D, G


def reset_gradients(model, require_grad):
    for p in model.parameters():
        p.requires_grad_(require_grad)
    return model
