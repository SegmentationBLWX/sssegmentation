'''
Function:
    Implementation of Logger
Author:
    Zhenchao Jin
'''
import logging


'''Logger'''
class Logger():
    def __init__(self, logfilepath):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)-8s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.FileHandler(logfilepath, encoding='utf-8'), logging.StreamHandler()]
        )
    '''log'''
    @staticmethod
    def log(level, message):
        logging.log(level, message)
    '''debug'''
    @staticmethod
    def debug(message):
        Logger.log(logging.DEBUG, message)
    '''info'''
    @staticmethod
    def info(message):
        Logger.log(logging.INFO, message)
    '''warning'''
    @staticmethod
    def warning(message):
        Logger.log(logging.WARNING, message)
    '''error'''
    @staticmethod
    def error(message):
        Logger.log(logging.ERROR, message)