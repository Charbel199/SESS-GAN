from conf import parse_args
from train.cascaded_train import CascadedModelTrainer
from logger.log import LoggerService
from generate_samples import generate_samples

logger = LoggerService.get_instance()


def main():
    config = parse_args()
    model_trainer = CascadedModelTrainer()
    generators, noise_maps, reals, noise_amplitudes = model_trainer.train_model(config=config)
    logger.info('Done training')
    generate_samples(generators, noise_maps, reals,
                     noise_amplitudes, config, input_image=None)


if __name__ == '__main__':
    main()
