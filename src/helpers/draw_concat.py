# Code based on https://github.com/tamarott/SinGAN
from torch.nn.functional import interpolate
from helpers.generate_noise import generate_spatial_noise
from conf import ModelConfig

def format_and_use_generator(curr_img, G_z, mode, Z_opt, pad_noise, pad_image, noise_amp, G):
    """ Correctly formats input for generator and runs it through. """
    if curr_img.shape != G_z.shape:
        G_z = interpolate(G_z, curr_img.shape[-2:], mode='bilinear', align_corners=False)

    if mode == "rand":
        curr_img = pad_noise(curr_img)  # Curr image is z in this case
        z_add = curr_img
    else:
        z_add = Z_opt
    G_z = pad_image(G_z)
    z_in = noise_amp * z_add + G_z
    G_z = G(z_in.detach(), G_z)
    return G_z


def draw_concat(generators, noise_maps, reals, noise_amplitudes, in_s, mode, pad_noise, pad_image, config: ModelConfig):
    """ Draw and concatenate output of the previous scale and a new noise map. """
    G_z = in_s
    if len(generators) > 0:
        if mode == "rand":
            noise_padding = 1 * config.number_of_layers
            for count, (G, Z_opt, real_curr, real_next, noise_amp) in enumerate(
                    zip(generators, noise_maps, reals, reals[1:], noise_amplitudes)):
                if count < config.num_of_scales:
                    z = generate_spatial_noise([1,
                                                real_curr.shape[1],
                                                Z_opt.shape[2] - 2 * noise_padding,
                                                Z_opt.shape[3] - 2 * noise_padding],
                                               device=config.device)
                G_z = format_and_use_generator(z, G_z, "rand", Z_opt,
                                               pad_noise, pad_image, noise_amp, G)

        if mode == "rec":
            for count, (G, Z_opt, real_curr, real_next, noise_amp) in enumerate(
                    zip(generators, noise_maps, reals, reals[1:], noise_amplitudes)):
                G_z = format_and_use_generator(real_curr, G_z, "rec", Z_opt,
                                               pad_noise, pad_image, noise_amp, G)

    return G_z
