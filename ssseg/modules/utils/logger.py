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
    def log(self, message, level='INFO', endwithnewline=True):
        message = f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} {level}  {message}'
        if not message.endswith('\n') and endwithnewline:
            message = message + '\n'
        print(message)
        self.fp_handler.write(message)
    '''debug'''
    def debug(self, message, endwithnewline=True):
        self.log(message, 'DEBUG', endwithnewline)
    '''info'''
    def info(self, message, endwithnewline=True):
        self.log(message, 'INFO', endwithnewline)
    '''warning'''
    def warning(self, message, endwithnewline=True):
        self.log(message, 'WARNING', endwithnewline)
    '''error'''
    def error(self, message, endwithnewline=True):
        self.log(message, 'ERROR', endwithnewline)
    '''close'''
    def close(self):
        self.fp_handler.close()