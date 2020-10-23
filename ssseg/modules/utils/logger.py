'''
Function:
    define the logging function
Author:
    Zhenchao Jin
'''
import logging


'''log function.'''
class Logger():
    def __init__(self, logfilepath, **kwargs):
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(levelname)-8s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            handlers=[logging.FileHandler(logfilepath),
                                      logging.StreamHandler()])
    @staticmethod
    def log(level, message):
        logging.log(level, message)
    @staticmethod
    def debug(message):
        Logger.log(logging.DEBUG, message)
    @staticmethod
    def info(message):
        Logger.log(logging.INFO, message)
    @staticmethod
    def warning(message):
        Logger.log(logging.WARNING, message)
    @staticmethod
    def error(message):
        Logger.log(logging.ERROR, message)