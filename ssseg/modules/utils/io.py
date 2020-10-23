'''
Function:
    some utils related with io
Author:
    Zhenchao Jin
'''
import os
import torch


'''check dir'''
def checkdir(dirname):
    if not os.path.exists(dirname):
        try: os.mkdir(dirname)
        except: pass
        return False
    return True


'''load checkpoints'''
def loadcheckpoints(checkpointspath, logger_handle=None, cmd_args=None):
    if logger_handle is not None and cmd_args.local_rank == 0:
        logger_handle.info('Loading checkpoints from %s...' % checkpointspath)
    checkpoints = torch.load(checkpointspath)
    return checkpoints


'''save checkpoints'''
def savecheckpoints(state_dict, savepath, logger_handle=None, cmd_args=None):
    if logger_handle is not None and cmd_args.local_rank == 0:
        logger_handle.info('Saving state_dict in %s...' % savepath)
    torch.save(state_dict, savepath)
    return True