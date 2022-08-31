import logging
import os
from dotenv import load_dotenv

load_dotenv()


class LoggerService(object):
    __instance = None

    def __init__(self):
        if self.__class__.__instance:
            raise Exception

        formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(str(os.getenv('LOG_FILE_PATH')))

        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger = logging.getLogger('SimulatorGAN')
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        self.__class__.__instance = logger

    @classmethod
    def get_instance(cls):
        if not cls.__instance:
            cls()

        return cls.__instance
