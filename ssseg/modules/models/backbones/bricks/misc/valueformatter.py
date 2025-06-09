'''
Function:
    Implementation of Value Formatter Functions
Author:
    Zhenchao Jin
'''
import numbers
import collections
import collections.abc


'''makedivisible'''
def makedivisible(value, divisor, min_value=None, min_ratio=0.9):
    if min_value is None: min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < min_ratio * value: new_value += divisor
    return new_value


'''totuple'''
def tolen2tuple(x):
    if isinstance(x, numbers.Number): return (x, x)
    assert isinstance(x, collections.abc.Sequence) and (len(x) == 2)
    for n in x: assert isinstance(n, numbers.Number)
    return tuple(x)