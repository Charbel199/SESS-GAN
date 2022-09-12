from conf import parse_args
from train.simple_train import ModelTrainer
from logger.log import LoggerService

logger = LoggerService.get_instance()


def main():
    conf = parse_args()
    model_trainer = ModelTrainer()
    model_trainer.train_model(config=conf)
    logger.info('Done training')


if __name__ == '__main__':
    main()
