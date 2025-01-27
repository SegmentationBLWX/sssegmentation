'''
Function:
    Implementation of Other Utils
Author:
    Zhenchao Jin
'''
import os
import torch
import random
import pickle
import numpy as np
import torch.distributed as dist
from .io import touchdirs


'''setrandomseed'''
def setrandomseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


'''ismainprocess'''
def ismainprocess():
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
    return str(os.environ['RANK']) in ['0']


'''postprocesspredgtpairs'''
def postprocesspredgtpairs(seg_results, cfg, logger_handle):
    # TODO: bug occurs if use --pyt bash in slurm
    process_id = int(os.environ['RANK'])
    # save results
    work_dir = os.path.join(cfg.SEGMENTOR_CFG['work_dir'], 'inference_local_results')
    touchdirs(work_dir)
    filename = f'seg_results_{process_id}.pkl'
    with open(os.path.join(work_dir, filename), 'wb') as fp:
        pickle.dump(seg_results, fp)
    process_id = torch.tensor([process_id], device='cuda')
    process_ids = [process_id.clone() for _ in range(int(os.environ['WORLD_SIZE']))]
    dist.all_gather(process_ids, process_id)
    logger_handle.info('Rank %s finished' % int(os.environ['RANK']))
    # post-process, here we assume that all nodes share a common storage space during multi-node training
    if int(os.environ['RANK']) == 0:
        seg_results_gather = {}
        for process_id in process_ids:
            process_id = str(int(process_id.item()))
            filename = f'seg_results_{process_id}.pkl'
            fp = open(os.path.join(work_dir, filename), 'rb')
            seg_results = pickle.load(fp)
            seg_results_gather.update(seg_results)
        seg_preds = [v['seg_pred'] for k, v in seg_results_gather.items()]
        seg_gts = [v['seg_gt'] for k, v in seg_results_gather.items()]
        seg_ids = [k for k, v in seg_results_gather.items()]
        logger_handle.info('All Finished, seg_preds: %s, seg_gts: %s' % (len(seg_preds), len(seg_gts)))
    else:
        seg_preds, seg_gts, seg_ids = [], [], []
    # return
    return seg_preds, seg_gts, seg_ids