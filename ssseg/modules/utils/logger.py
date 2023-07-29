'''
Function:
    Implementation of Logger
Author:
    Zhenchao Jin
'''
import os
import time


'''Logger'''
class Logger():
    def __init__(self, logfilepath):
        self.logfilepath = logfilepath
        self.fp_handler = open(logfilepath, 'a')
    '''log'''
    def log(self, message, level='INFO'):
        message = f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} {level} {message}'
        print(message)
        self.fp_handler.write(message)
    '''debug'''
    def debug(self, message):
        self.log(message, 'DEBUG')
    '''info'''
    def info(self, message):
        self.log(message, 'INFO')
    '''warning'''
    def warning(self, message):
        self.log(message, 'WARNING')
    '''error'''
    def error(self, message):
        self.log(message, 'ERROR')
    '''close'''
    def close(self):
        self.fp_handler.close()