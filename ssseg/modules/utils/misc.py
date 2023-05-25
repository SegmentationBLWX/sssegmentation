'''
Function:
    Implementation of Utils
Author:
    Zhenchao Jin
'''
import torch
import random
import numpy as np


'''setrandomseed'''
def setrandomseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)