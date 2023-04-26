'''
Function:
    Implementation of SLURM related operations
Author:
    Zhenchao Jin
'''
import os
import torch
import subprocess


'''initslurm'''
def initslurm(cmd_args, master_port='29000'):
    if 'SLURM_PROCID' not in os.environ:
        return False
    rank = int(os.environ['SLURM_PROCID'])
    gpu = rank % torch.cuda.device_count()
    world_size = int(os.environ['SLURM_NTASKS'])
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = master_port
    node_list = os.environ['SLURM_NODELIST']
    addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = addr
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(gpu)
    os.environ['WORLD_SIZE'] = str(world_size)
    cmd_args.local_rank = int(os.environ['LOCAL_RANK'])
    return True