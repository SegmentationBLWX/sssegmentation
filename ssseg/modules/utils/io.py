'''
Function:
    Implementation of IO related operations
Author:
    Zhenchao Jin
'''
import os
import torch


'''touchdir'''
def touchdir(dirname):
    if not os.path.exists(dirname):
        try: os.mkdir(dirname)
        except: pass
        return False
    return True


'''loadckpts'''
def loadckpts(ckptspath, map_to_cpu=True):
    if map_to_cpu: 
        ckpts = torch.load(ckptspath, map_location=torch.device('cpu'))
    else: 
        ckpts = torch.load(ckptspath)
    return ckpts


'''saveckpts'''
def saveckpts(ckpts, savepath):
    save_response = torch.save(ckpts, savepath)
    return save_response