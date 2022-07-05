import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Run SimulatorGAN')

    parser.add_argument('-b', '--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('-e', '--epoch', type=int, default=10, help='Number of epochs')
    parser.add_argument('-d', '--device', type=str, default='cpu', help='Device (cpu, gpu ...)')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()


if __name__ == '__main__':
    main()
