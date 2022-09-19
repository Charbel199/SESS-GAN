from conf import parse_args
from train.cascaded_train import CascadedModelTrainer
from logger.log import LoggerService

logger = LoggerService.get_instance()


def main():
    conf = parse_args()
    model_trainer = CascadedModelTrainer()
    model_trainer.train_model(config=conf)
    logger.info('Done training')


if __name__ == '__main__':
    main()
