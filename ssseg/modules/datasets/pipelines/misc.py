'''
Function:
    Implementation of pipelines-related functions
Author:
    Zhenchao Jin
'''
import numbers
import collections
import collections.abc


'''assertvalidprob'''
def assertvalidprob(prob):
    assert isinstance(prob, numbers.Number), 'invalid prob, data type should be set as numbers.Number'
    assert prob >= 0.0 and prob <= 1.0, 'invalid prob, data range should be in [0, 1]'


'''assertvalidimagesize'''
def assertvalidimagesize(image_size):
    assert isinstance(image_size, int) or (isinstance(image_size, collections.abc.Sequence) and len(image_size) == 2), 'invalid image size, data type should be set as int or collections.abc.Sequence with length equal to 2'


'''assertvalidrange'''
def assertvalidrange(x_range, allow_none=False, allow_number=False):
    if allow_number and isinstance(x_range, numbers.Number):
        return
    if allow_none:
        assert (isinstance(x_range, collections.abc.Sequence) and len(x_range) == 2) or x_range is None, 'invalid x_range, x_range should be set as [low, high] or None'
    else:
        assert isinstance(x_range, collections.abc.Sequence) and len(x_range) == 2, 'invalid x_range, x_range should be set as [low, high]'
    if x_range is not None:
        for n in x_range: assert isinstance(n, numbers.Number), 'invalid x_range, data_type in x_range should be set as numbers.Number'
        assert x_range[1] >= x_range[0], 'invalid x_range, x_range should be set as [low, high] and high >= low'


'''totuple'''
def totuple(x):
    if isinstance(x, numbers.Number): return (x, x)
    assert isinstance(x, collections.abc.Sequence) and (len(x) == 2)
    for n in x: assert isinstance(n, numbers.Number)
    return tuple(x)