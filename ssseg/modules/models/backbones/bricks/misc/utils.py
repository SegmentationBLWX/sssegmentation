'''
Function:
    Implementation of some utils
Author:
    Zhenchao Jin
'''
import torch


'''makedivisible'''
def makedivisible(value, divisor, min_value=None, min_ratio=0.9):
    if min_value is None: min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < min_ratio * value: new_value += divisor
    return new_value