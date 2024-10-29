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
        initslurm, touchdirs, loadckpts, postprocesspredgtpairs, BuildDistributedDataloader, BuildDistributedModel, ismainprocess,
        BuildDataset, BuildSegmentor, BuildLoggerHandle, ConfigParser
    )
except:
    from .modules import (
        initslurm, touchdirs, loadckpts, postprocesspredgtpairs, BuildDistributedDataloader, BuildDistributedModel, ismainprocess,
        BuildDataset, BuildSegmentor, BuildLoggerHandle, ConfigParser
    )
warnings.filterwarnings('ignore')


'''parsecmdargs'''
def parsecmdargs():
    parser = argparse.ArgumentParser(description='SSSegmentation is an open source supervised semantic segmentation toolbox based on PyTorch')
    parser.add_argument('--local_rank', '--local-rank', dest='local_rank', help='node rank for distributed testing', default=0, type=int)
    parser.add_argument('--nproc_per_node', dest='nproc_per_node', help='number of process per node', default=8, type=int)
    parser.add_argument('--cfgfilepath', dest='cfgfilepath', help='config file path you want to use', type=str, required=True)
    parser.add_argument('--eval_env', dest='eval_env', help='evaluate environment, support server and local', default='local', type=str, choices=['server', 'local'])
    parser.add_argument('--ckptspath', dest='ckptspath', help='checkpoints you want to resume from', type=str, required=True)
    parser.add_argument('--slurm', dest='slurm', help='please add --slurm if you are using slurm', default=False, action='store_true')
    parser.add_argument('--ema', dest='ema', help='please add --ema if you want to load ema weights for segmentors', default=False, action='store_true')
    cmd_args = parser.parse_args()
    if torch.__version__.startswith('2.'):
        cmd_args.local_rank = int(os.environ['LOCAL_RANK'])
    if cmd_args.slurm:
        initslurm(cmd_args, str(6666 + random.randint(0, 1000)))
    else:
        if 'RANK' not in os.environ:
            os.environ['RANK'] = str(cmd_args.local_rank)
        if 'LOCAL_RANK' not in os.environ:
            os.environ['LOCAL_RANK'] = str(cmd_args.local_rank)
    return cmd_args


'''Tester'''
class Tester():
    def __init__(self, cmd_args):
        # parse config file
        cfg, cfg_file_path = ConfigParser()(cmd_args.cfgfilepath)
        # touch work dir
        touchdirs(cfg.SEGMENTOR_CFG['work_dir'])
        # initialize logger_handle
        logger_handle = BuildLoggerHandle(cfg.SEGMENTOR_CFG['logger_handle_cfg'])
        # number of gpus per node, for distribued testing, only support a process for a GPU
        ngpus_per_node = torch.cuda.device_count()
        if ngpus_per_node != cmd_args.nproc_per_node:
            logger_handle.warning('ngpus_per_node is not equal to nproc_per_node, force ngpus_per_node = nproc_per_node by default', main_process_only=True)
            ngpus_per_node = cmd_args.nproc_per_node
        # set attributes
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
    def start(self):
        # initialize necessary variables
        cfg, logger_handle, cmd_args, cfg_file_path = self.cfg, self.logger_handle, self.cmd_args, self.cfg_file_path
        rank_id = int(os.environ['RANK'])
        # build dataset and dataloader
        cfg.SEGMENTOR_CFG['dataset']['test']['eval_env'] = self.cmd_args.eval_env
        dataset = BuildDataset(mode='TEST', logger_handle=logger_handle, dataset_cfg=cfg.SEGMENTOR_CFG['dataset'])
        assert dataset.num_classes == cfg.SEGMENTOR_CFG['num_classes'], 'parsed config file %s error' % cfg_file_path
        dataloader_cfg = copy.deepcopy(cfg.SEGMENTOR_CFG['dataloader'])
        dataloader_cfg['test']['batch_size'], dataloader_cfg['test']['num_workers'] = dataloader_cfg['test'].pop('batch_size_per_gpu'), dataloader_cfg['test'].pop('num_workers_per_gpu')
        dataloader = BuildDistributedDataloader(dataset=dataset, dataloader_cfg=dataloader_cfg['test'])
        # build segmentor
        cfg.SEGMENTOR_CFG['backbone']['pretrained'] = False
        segmentor = BuildSegmentor(segmentor_cfg=cfg.SEGMENTOR_CFG, mode='TEST')
        torch.cuda.set_device(cmd_args.local_rank)
        segmentor.cuda(cmd_args.local_rank)
        # load ckpts
        ckpts = loadckpts(cmd_args.ckptspath)
        try:
            segmentor.load_state_dict(ckpts['model'] if not cmd_args.ema else ckpts['model_ema'])
        except Exception as e:
            logger_handle.warning(str(e) + '\n' + 'Try to load ckpts by using strict=False', main_process_only=True)
            segmentor.load_state_dict(ckpts['model'] if not cmd_args.ema else ckpts['model_ema'], strict=False)
        # parallel
        segmentor = BuildDistributedModel(segmentor, {'device_ids': [cmd_args.local_rank]})
        # print information
        logger_handle.info(f'Config file path: {cfg_file_path}', main_process_only=True)
        logger_handle.info(f'Config details: \n{cfg.SEGMENTOR_CFG}', main_process_only=True)
        logger_handle.info(f'Resume from: {cmd_args.ckptspath}', main_process_only=True)
        # set eval
        segmentor.eval()
        # start to test
        inference_cfg, seg_results = copy.deepcopy(cfg.SEGMENTOR_CFG['inference']), {}
        with torch.no_grad():
            dataloader.sampler.set_epoch(0)
            pbar = tqdm(enumerate(dataloader))
            for batch_idx, samples_meta in pbar:
                pbar.set_description('Processing %s/%s in rank %s' % (batch_idx+1, len(dataloader), rank_id))
                infer_tta_cfg, align_corners = inference_cfg['tta'], segmentor.module.align_corners
                cascade_cfg = infer_tta_cfg.get('cascade', {'key_for_pre_output': 'memory_gather_logits', 'times': 1, 'forward_default_args': None})
                for time_idx in range(cascade_cfg['times']):
                    forward_args = None
                    if time_idx > 0: 
                        seg_logits_list = [F.interpolate(seg_logits, size=seg_logits_list[-1].shape[2:], mode='bilinear', align_corners=align_corners) for seg_logits in seg_logits_list]
                        forward_args = {cascade_cfg['key_for_pre_output']: sum(seg_logits_list) / len(seg_logits_list)}
                        if cascade_cfg['forward_default_args'] is not None:
                            forward_args.update(cascade_cfg['forward_default_args'])
                    seg_logits_list = segmentor.module.auginference(samples_meta['image'], forward_args)
                for seg_idx in range(len(seg_logits_list[0])):
                    seg_logit_list = [F.interpolate(seg_logits[seg_idx: seg_idx+1], size=(samples_meta['height'][seg_idx], samples_meta['width'][seg_idx]), mode='bilinear', align_corners=align_corners) for seg_logits in seg_logits_list]
                    seg_logit = sum(seg_logit_list) / len(seg_logit_list)
                    seg_pred = (torch.argmax(seg_logit[0], dim=0)).cpu().numpy().astype(np.int32)
                    seg_gt = samples_meta['seg_target'][seg_idx].cpu().numpy().astype(np.int32)
                    seg_gt[seg_gt >= dataset.num_classes] = -1
                    seg_results[samples_meta['id'][seg_idx]] = {'seg_pred': seg_pred, 'seg_gt': seg_gt}
        # post process
        seg_preds, seg_gts, seg_ids = postprocesspredgtpairs(seg_results=seg_results, cmd_args=cmd_args, cfg=cfg, logger_handle=logger_handle)
        # evaluate
        if ismainprocess():
            dataset = BuildDataset(mode='TEST', logger_handle=logger_handle, dataset_cfg=cfg.SEGMENTOR_CFG['dataset'])
            if cmd_args.eval_env == 'local':
                result = dataset.evaluate(
                    seg_preds=seg_preds, seg_gts=seg_gts, num_classes=cfg.SEGMENTOR_CFG['num_classes'], ignore_index=-1, **cfg.SEGMENTOR_CFG['inference'].get('evaluate', {}),
                )
                logger_handle.info(result, main_process_only=True)
            else:
                dataset.formatresults(seg_preds, seg_ids, savedir=os.path.join(cfg.SEGMENTOR_CFG['work_dir'], 'inference_server_results'))


'''run'''
if __name__ == '__main__':
    with torch.no_grad():
        # parse arguments
        cmd_args = parsecmdargs()
        # instanced Tester
        client = Tester(cmd_args=cmd_args)
        # start
        client.start()