'''
Function:
    Implementation of Tester
Author:
    Zhenchao Jin
'''
import os
import copy
import torch
import random
import warnings
import argparse
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
try:
    from modules import (
        initslurm, touchdir, loadckpts, postprocesspredgtpairs, BuildDistributedDataloader, BuildDistributedModel,
        BuildDataset, BuildSegmentor, BuildLoggerHandle, ConfigParser
    )
except:
    from .modules import (
        initslurm, touchdir, loadckpts, postprocesspredgtpairs, BuildDistributedDataloader, BuildDistributedModel,
        BuildDataset, BuildSegmentor, BuildLoggerHandle, ConfigParser
    )
warnings.filterwarnings('ignore')


'''parsecmdargs'''
def parsecmdargs():
    parser = argparse.ArgumentParser(description='SSSegmentation is an open source supervised semantic segmentation toolbox based on PyTorch')
    parser.add_argument('--local_rank', '--local-rank', dest='local_rank', help='node rank for distributed testing', default=0, type=int)
    parser.add_argument('--nproc_per_node', dest='nproc_per_node', help='number of process per node', default=8, type=int)
    parser.add_argument('--cfgfilepath', dest='cfgfilepath', help='config file path you want to use', type=str, required=True)
    parser.add_argument('--evalmode', dest='evalmode', help='evaluate mode, support server and local', default='local', type=str, choices=['server', 'local'])
    parser.add_argument('--ckptspath', dest='ckptspath', help='checkpoints you want to resume from', type=str, required=True)
    parser.add_argument('--slurm', dest='slurm', help='please add --slurm if you are using slurm', default=False, action='store_true')
    parser.add_argument('--ema', dest='ema', help='please add --ema if you want to load ema weights for segmentors', default=False, action='store_true')
    args = parser.parse_args()
    if torch.__version__.startswith('2.'):
        args.local_rank = int(os.environ['LOCAL_RANK'])
    if args.slurm: initslurm(args, str(6666 + random.randint(0, 1000)))
    return args


'''Tester'''
class Tester():
    def __init__(self, cfg, ngpus_per_node, logger_handle, cmd_args, cfg_file_path):
        # set attribute
        self.cfg = cfg
        self.ngpus_per_node = ngpus_per_node
        self.logger_handle = logger_handle
        self.cmd_args = cmd_args
        self.cfg_file_path = cfg_file_path
        assert torch.cuda.is_available(), 'cuda is not available'
        # init distributed training
        dist.init_process_group(backend=self.cfg.SEGMENTOR_CFG.get('backend', 'nccl'))
        # open full fp32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    '''start tester'''
    def start(self, all_preds, all_gts):
        cfg, ngpus_per_node, logger_handle, cmd_args, cfg_file_path = self.cfg, self.ngpus_per_node, self.logger_handle, self.cmd_args, self.cfg_file_path
        rank_id = int(os.environ['SLURM_PROCID']) if 'SLURM_PROCID' in os.environ else cmd_args.local_rank
        # build dataset and dataloader
        cfg.SEGMENTOR_CFG['dataset']['evalmode'] = self.cmd_args.evalmode
        dataset = BuildDataset(mode='TEST', logger_handle=logger_handle, dataset_cfg=cfg.SEGMENTOR_CFG['dataset'])
        assert dataset.num_classes == cfg.SEGMENTOR_CFG['num_classes'], 'parsed config file %s error' % cfg_file_path
        dataloader_cfg = copy.deepcopy(cfg.SEGMENTOR_CFG['dataloader'])
        dataloader_cfg['test']['batch_size'], dataloader_cfg['test']['num_workers'] = dataloader_cfg['test'].pop('batch_size_per_gpu'), dataloader_cfg['test'].pop('num_workers_per_gpu')
        dataloader = BuildDistributedDataloader(dataset=dataset, dataloader_cfg=dataloader_cfg['test'])
        # build segmentor
        cfg.SEGMENTOR_CFG['backbone']['pretrained'] = False
        segmentor = BuildSegmentor(segmentor_cfg=copy.deepcopy(cfg.SEGMENTOR_CFG), mode='TEST')
        torch.cuda.set_device(cmd_args.local_rank)
        segmentor.cuda(cmd_args.local_rank)
        # load ckpts
        ckpts = loadckpts(cmd_args.ckptspath)
        try:
            segmentor.load_state_dict(ckpts['model'] if not cmd_args.ema else ckpts['model_ema'])
        except Exception as e:
            logger_handle.warning(str(e) + '\n' + 'Try to load ckpts by using strict=False')
            segmentor.load_state_dict(ckpts['model'] if not cmd_args.ema else ckpts['model_ema'], strict=False)
        # parallel
        segmentor = BuildDistributedModel(segmentor, {'device_ids': [cmd_args.local_rank]})
        # print information
        if (cmd_args.local_rank == 0) and (int(os.environ.get('SLURM_PROCID', 0)) == 0):
            logger_handle.info(f'Config file path: {cfg_file_path}')
            logger_handle.info(f'Config details: \n{cfg.SEGMENTOR_CFG}')
            logger_handle.info(f'Resume from: {cmd_args.ckptspath}')
        # set eval
        segmentor.eval()
        # start to test
        inference_cfg = copy.deepcopy(cfg.SEGMENTOR_CFG['inference'])
        with torch.no_grad():
            dataloader.sampler.set_epoch(0)
            pbar = tqdm(enumerate(dataloader))
            for batch_idx, samples_meta in pbar:
                pbar.set_description('Processing %s/%s in rank %s' % (batch_idx+1, len(dataloader), rank_id))
                imageids, images, widths, heights, gts = samples_meta['id'], samples_meta['image'], samples_meta['width'], samples_meta['height'], samples_meta['seg_target']
                infer_tricks, align_corners = inference_cfg['tricks'], segmentor.module.align_corners
                cascade_cfg = infer_tricks.get('cascade', {'key_for_pre_output': 'memory_gather_logits', 'times': 1, 'forward_default_args': None})
                for idx in range(cascade_cfg['times']):
                    forward_args = None
                    if idx > 0: 
                        outputs_list = [F.interpolate(outputs, size=outputs_list[-1].shape[2:], mode='bilinear', align_corners=align_corners) for outputs in outputs_list]
                        forward_args = {cascade_cfg['key_for_pre_output']: sum(outputs_list) / len(outputs_list)}
                        if cascade_cfg['forward_default_args'] is not None: 
                            forward_args.update(cascade_cfg['forward_default_args'])
                    outputs_list = segmentor.module.auginference(images, forward_args)
                for idx in range(len(outputs_list[0])):
                    output = [F.interpolate(outputs[idx: idx+1], size=(heights[idx], widths[idx]), mode='bilinear', align_corners=align_corners) for outputs in outputs_list]
                    output = sum(output) / len(output)
                    pred = (torch.argmax(output[0], dim=0)).cpu().numpy().astype(np.int32)
                    all_preds.append([imageids[idx], pred])
                    gt = gts[idx].cpu().numpy().astype(np.int32)
                    gt[gt >= dataset.num_classes] = -1
                    all_gts.append(gt)


'''main'''
def main():
    # parse arguments
    args = parsecmdargs()
    cfg, cfg_file_path = ConfigParser()(args.cfgfilepath)
    # touch work dir
    touchdir(cfg.SEGMENTOR_CFG['work_dir'])
    # initialize logger_handle
    logger_handle = BuildLoggerHandle(cfg.SEGMENTOR_CFG['logger_handle_cfg'])
    # number of gpus, for distribued testing, only support a process for a GPU
    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node != args.nproc_per_node:
        if (args.local_rank == 0) and (int(os.environ.get('SLURM_PROCID', 0)) == 0):
            logger_handle.warning('ngpus_per_node is not equal to nproc_per_node, force ngpus_per_node = nproc_per_node by default')
        ngpus_per_node = args.nproc_per_node
    # instanced Tester
    all_preds, all_gts = [], []
    client = Tester(cfg=cfg, ngpus_per_node=ngpus_per_node, logger_handle=logger_handle, cmd_args=args, cfg_file_path=cfg_file_path)
    client.start(all_preds, all_gts)
    # post process
    all_preds, all_gts, all_ids = postprocesspredgtpairs(all_preds=all_preds, all_gts=all_gts, cmd_args=args, cfg=cfg, logger_handle=logger_handle)
    # evaluate
    if (args.local_rank == 0) and (int(os.environ.get('SLURM_PROCID', 0)) == 0):
        dataset = BuildDataset(mode='TEST', logger_handle=logger_handle, dataset_cfg=cfg.SEGMENTOR_CFG['dataset'])
        if args.evalmode == 'local':
            result = dataset.evaluate(
                seg_preds=all_preds, seg_targets=all_gts, metric_list=cfg.SEGMENTOR_CFG['inference'].get('metric_list', ['iou', 'miou']),
                num_classes=cfg.SEGMENTOR_CFG['num_classes'], ignore_index=-1,
            )
            logger_handle.info(result)
        else:
            dataset.formatresults(all_preds, all_ids, savedir=os.path.join(cfg.SEGMENTOR_CFG['work_dir'], 'results'))


'''debug'''
if __name__ == '__main__':
    with torch.no_grad():
        main()