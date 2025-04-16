from conf import parse_args
from logger.log import LoggerService
from model.singan.models_helper import load_trained_components
from evaluate.generate_samples import generate_map
import os
logger = LoggerService.get_instance()


def main():
    config = parse_args()
    generators_m, noise_maps_m, reals_m, noise_amplitudes_m = load_trained_components(config)
    num_samples = 100
    generate_map(config, num_samples=num_samples, generators=generators_m, save = True, save_dir=os.path.join(config.output_path,'samples'),generator_index=1)

if __name__ == '__main__':
    main()
