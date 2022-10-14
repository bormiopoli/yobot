import logging
import os

def initialize_logger(logging, root):
    logger = logging.getLogger('Yo-BOT_log')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(f'{root}/YoBOT_program.log', mode='w')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

root = os.getcwd()

logger = initialize_logger(logging, root)
