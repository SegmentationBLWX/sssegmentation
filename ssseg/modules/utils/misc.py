'''
Function:
    Implementation of Utils
Author:
    Zhenchao Jin
'''
import os
import torch
import random
import pickle
import numpy as np
import torch.distributed as dist


'''setrandomseed'''
def setrandomseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


'''postprocesspredgtpairs'''
def postprocesspredgtpairs(seg_preds, seg_gts, cmd_args, cfg, logger_handle):
    # TODO: bug occurs if use --pyt bash in slurm
    rank_id = int(os.environ['SLURM_PROCID']) if 'SLURM_PROCID' in os.environ else cmd_args.local_rank
    # save results
    work_dir = cfg.SEGMENTOR_CFG['work_dir']
    filename = cfg.SEGMENTOR_CFG['evaluate_results_filename'].split('.')[0] + f'_{rank_id}.' + cfg.SEGMENTOR_CFG['evaluate_results_filename'].split('.')[-1]
    with open(os.path.join(work_dir, filename), 'wb') as fp:
        pickle.dump([seg_preds, seg_gts], fp)
    rank = torch.tensor([rank_id], device='cuda')
    rank_list = [rank.clone() for _ in range(cmd_args.nproc_per_node)]
    dist.all_gather(rank_list, rank)
    logger_handle.info('Rank %s finished' % int(rank.item()))
    # post-process
    if (cmd_args.local_rank == 0) and (int(os.environ.get('SLURM_PROCID', 0)) == 0):
        seg_preds_gather, seg_gts_gather = [], []
        for rank in rank_list:
            rank = str(int(rank.item()))
            filename = cfg.SEGMENTOR_CFG['evaluate_results_filename'].split('.')[0] + f'_{rank}.' + cfg.SEGMENTOR_CFG['evaluate_results_filename'].split('.')[-1]
            fp = open(os.path.join(work_dir, filename), 'rb')
            seg_preds, seg_gts = pickle.load(fp)
            seg_preds_gather += seg_preds
            seg_gts_gather += seg_gts
        seg_preds, seg_gts = seg_preds_gather, seg_gts_gather
        seg_preds_filtered, seg_gts_filtered, seg_ids = [], [], []
        for idx, pred in enumerate(seg_preds):
            if pred[0] in seg_ids:
                continue
            seg_ids.append(pred[0])
            seg_preds_filtered.append(pred[1])
            seg_gts_filtered.append(seg_gts[idx])
        seg_preds, seg_gts = seg_preds_filtered, seg_gts_filtered
        logger_handle.info('All Finished, seg_preds: %s, seg_gts: %s' % (len(seg_preds), len(seg_gts)))
    else:
        seg_preds, seg_gts, seg_ids = [], [], []
    # return
    return seg_preds, seg_gts, seg_ids