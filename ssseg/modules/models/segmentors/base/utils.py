'''
Function:
    Implementation of Utils
Author:
    Zhenchao Jin
'''
import torch


'''attrfetcher'''
def attrfetcher(obj, attribute):
    if '.' in attribute:
        for attr in attribute.split('.'):
            obj = getattr(obj, attr)
    else:
        obj = getattr(obj, attribute)
    return obj


'''attrjudger'''
def attrjudger(obj, attribute):
    if '.' in attribute:
        for attr in attribute.split('.'):
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                return False
        return True
    else:
        return hasattr(obj, attribute)