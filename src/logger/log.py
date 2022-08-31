import logging
import os
from dotenv import load_dotenv

load_dotenv()


def setup_custom_logger():
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(str(os.getenv('LOG_FILE_PATH')))

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger = logging.getLogger('SimulatorGAN')
    logger.setLevel(logging.DEBUG)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger
