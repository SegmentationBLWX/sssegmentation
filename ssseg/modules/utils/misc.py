'''
Function:
    Define some utils
Author:
    Zhenchao Jin
'''
import torch
import random
import numpy as np


'''setRandomSeed'''
def setRandomSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)