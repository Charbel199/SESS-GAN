from conf import parse_args
from logger.log import LoggerService
from model.singan.models_helper import load_trained_components
from genetic.genetic_algorithm import GeneticAlgorithm

logger = LoggerService.get_instance()


def main():
    config = parse_args()
    generators_m, noise_maps_m, reals_m, noise_amplitudes_m = load_trained_components(config)
    genetic_algorithm = GeneticAlgorithm(config, generators_m, population_size=30, remaining_population_percentage=0.2)
    genetic_algorithm.execute(generations=50, threshold=0.98)


if __name__ == '__main__':
    main()
