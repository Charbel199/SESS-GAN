import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from model.discriminator import Discriminator
from model.generator import Generator
from model.weights import initialize_model_weights
from model.preprocessing import transforms


def process_args(args):
    if torch.cuda.is_available() and args.device == 'cpu':
        print("WARNING: You have a CUDA device, so you should probably run with -d cuda:0")

    return args


def parse_args():
    parser = argparse.ArgumentParser(description='Run SimulatorGAN')

    parser.add_argument('-b', '--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('-e', '--epoch', type=int, default=10, help='Number of epochs')
    parser.add_argument('-d', '--device', type=str, default='cpu', help='Device (cpu, cuda:0 ...)')
    parser.add_argument('-s', '--size', type=int, default=50, help='Image size')
    parser.add_argument('-lr', '--learning-rate', type=float, default=3e-4, help='Learning Rate')
    parser.add_argument('-nd', '--noise-dimension', type=int, default=100, help='Noise dimension')
    parser.add_argument('-nc', '--number-of-classes', type=int, default=3, help='Number of classes')
    parser.add_argument('-fd', '--features_discriminator', type=int, default=64,
                        help='Scaling discriminator features factor')
    parser.add_argument('-fd', '--features_generator', type=int, default=64, help='Scaling generator features factor')

    args = parser.parse_args()
    args = process_args(args)

    return args


def main():
    args = parse_args()

    fixed_noise = torch.randn(args.batch_size, args.noise_dimension, 1, 1).to(args.device)
    writer_fake = SummaryWriter(f"assets/logs/runs/fake")  # For fake images
    writer_real = SummaryWriter(f"assets/logs/runs/real")  # For real images

    discriminator = Discriminator(args.number_of_classes, args.features_discriminator).to(args.device)
    initialize_model_weights(discriminator)
    generator = Generator(args.noise_dimension, args.number_of_classes, args.features_generator).to(args.device)
    initialize_model_weights(generator)


if __name__ == '__main__':
    main()
