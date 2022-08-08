'''
Function:
    Some utils related with io
Author:
    Zhenchao Jin
'''
import os
import torch


'''checkdir'''
def checkdir(dirname):
    if not os.path.exists(dirname):
        try: os.mkdir(dirname)
        except: pass
        return False
    return True


'''loadcheckpoints'''
def loadcheckpoints(checkpointspath, logger_handle=None, cmd_args=None, map_to_cpu=True):
    if (logger_handle is not None) and (cmd_args is None or cmd_args.local_rank == 0) and (int(os.environ.get('SLURM_PROCID', 0)) == 0):
        logger_handle.info('Loading checkpoints from %s' % checkpointspath)
    if map_to_cpu: checkpoints = torch.load(checkpointspath, map_location=torch.device('cpu'))
    else: checkpoints = torch.load(checkpointspath)
    return checkpoints


'''savecheckpoints'''
def savecheckpoints(state_dict, savepath, logger_handle=None, cmd_args=None):
    if (logger_handle is not None) and (cmd_args is None or cmd_args.local_rank == 0) and (int(os.environ.get('SLURM_PROCID', 0)) == 0):
        logger_handle.info('Saving state_dict in %s' % savepath)
    torch.save(state_dict, savepath)
    return True