from conf import parse_args
from train import train_model
from logger.log import LoggerService

logger = LoggerService.get_instance()


def main():
    conf = parse_args()
    train_model(config=conf)
    logger.info('Done training')


if __name__ == '__main__':
    main()
