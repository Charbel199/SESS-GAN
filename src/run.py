import argparse
import torch
from conf import ModelConfig
from train import train_model
import logger.log as log
logger = log.setup_custom_logger()

def process_args(args) -> ModelConfig:
    if torch.cuda.is_available() and args.device == 'cpu':
        print("WARNING: You have a CUDA device, so you should probably run with -d cuda:0")
    config: ModelConfig = ModelConfig(args)
    return config


def parse_args() -> ModelConfig:
    parser = argparse.ArgumentParser(description='Run SimulatorGAN')

    parser.add_argument('-b', '--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('-e', '--epoch', type=int, default=10, help='Number of epochs')
    parser.add_argument('-d', '--device', type=str, default='cpu', help='Device (cpu, cuda:0 ...)')
    parser.add_argument('-s', '--size', type=int, default=50, help='Image size')
    parser.add_argument('-lr', '--learning-rate', type=float, default=3e-4, help='Learning Rate')
    parser.add_argument('-nd', '--noise-dimension', type=int, default=100, help='Noise dimension')
    parser.add_argument('-nc', '--number-of-classes', type=int, default=3, help='Number of classes')
    parser.add_argument('-fd', '--features-discriminator', type=int, default=64,
                        help='Scaling discriminator features factor')
    parser.add_argument('-fg', '--features-generator', type=int, default=64,
                        help='Scaling generator features factor')
    parser.add_argument('-tp', '--train-path', type=str, required=True, help='Training data path')
    parser.add_argument('-vp', '--val-path', type=str, required=True, help='Validation data path')
    parser.add_argument('-df', '--data-format', type=str, default='txt', help='Data type (txt, csv ...)')
    parser.add_argument('-ks', '--kernel-size', type=int, default='3', help='Convolution kernel size')

    args = parser.parse_args()
    conf = process_args(args)

    return conf


def main():
    conf = parse_args()
    train_model(config=conf)
    logger.info('Done training')


if __name__ == '__main__':
    main()
