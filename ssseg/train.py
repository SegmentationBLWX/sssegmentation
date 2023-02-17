'''
Function:
    Train the segmentor
Author:
    Zhenchao Jin
'''
import os
import copy
import torch
import pickle
import warnings
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
from configs import BuildConfig
from modules import (
    BuildDataset, BuildDistributedDataloader, BuildDistributedModel, BuildOptimizer, BuildScheduler, initslurm,
    BuildLoss, BuildBackbone, BuildSegmentor, BuildPixelSampler, Logger, setRandomSeed, BuildPalette, checkdir, loadcheckpoints, savecheckpoints
)
warnings.filterwarnings('ignore')


'''parse arguments in command line'''
def parseArgs():
    parser = argparse.ArgumentParser(description='SSSegmentation is an open source supervised semantic segmentation toolbox based on PyTorch')
    parser.add_argument('--local_rank', dest='local_rank', help='node rank for distributed training', default=0, type=int)
    parser.add_argument('--nproc_per_node', dest='nproc_per_node', help='number of process per node', default=8, type=int)
    parser.add_argument('--cfgfilepath', dest='cfgfilepath', help='config file path you want to use', type=str, required=True)
    parser.add_argument('--checkpointspath', dest='checkpointspath', help='checkpoints you want to resume from', default='', type=str)
    parser.add_argument('--slurm', dest='slurm', help='please add --slurm if you are using slurm', default=False, action='store_true')
    args = parser.parse_args()
    if args.slurm: initslurm(args, '29000')
    return args


'''Trainer'''
class Trainer():
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
    '''start trainer'''
    def start(self):
        cfg, ngpus_per_node, logger_handle, cmd_args, cfg_file_path = self.cfg, self.ngpus_per_node, self.logger_handle, self.cmd_args, self.cfg_file_path
        # build dataset and dataloader
        dataset = BuildDataset(mode='TRAIN', logger_handle=logger_handle, dataset_cfg=copy.deepcopy(cfg.DATASET_CFG))
        assert dataset.num_classes == cfg.SEGMENTOR_CFG['num_classes'], 'parsed config file %s error' % cfg_file_path
        dataloader_cfg = copy.deepcopy(cfg.DATALOADER_CFG)
        batch_size, num_workers = dataloader_cfg['train']['batch_size'], dataloader_cfg['train']['num_workers']
        batch_size_per_node = batch_size // ngpus_per_node
        num_workers_per_node = num_workers // ngpus_per_node
        dataloader_cfg['train'].update({'batch_size': batch_size_per_node, 'num_workers': num_workers_per_node})
        dataloader = BuildDistributedDataloader(dataset=dataset, dataloader_cfg=dataloader_cfg['train'])
        # build segmentor
        segmentor = BuildSegmentor(segmentor_cfg=copy.deepcopy(cfg.SEGMENTOR_CFG), mode='TRAIN')
        torch.cuda.set_device(cmd_args.local_rank)
        segmentor.cuda(cmd_args.local_rank)
        torch.backends.cudnn.benchmark = cfg.SEGMENTOR_CFG['benchmark']
        # build optimizer
        optimizer_cfg = copy.deepcopy(cfg.OPTIMIZER_CFG)
        optimizer = BuildOptimizer(segmentor, optimizer_cfg)
        # build fp16
        fp16_cfg = self.cfg.SEGMENTOR_CFG.get('fp16_cfg', {'type': None})
        fp16_type = fp16_cfg.pop('type')
        assert fp16_type in [None, 'apex']
        if fp16_type is not None:
            import apex
            segmentor, optimizer = apex.amp.initialize(segmentor, optimizer, **fp16_cfg)
            for m in segmentor.modules():
                if hasattr(m, 'fp16_enabled'):
                    m.fp16_enabled = True
        # build scheduler
        scheduler_cfg = copy.deepcopy(cfg.SCHEDULER_CFG)
        scheduler_cfg.update({
            'lr': cfg.OPTIMIZER_CFG['lr'],
            'iters_per_epoch': len(dataloader),
            'params_rules': cfg.OPTIMIZER_CFG['params_rules'],
        })
        scheduler = BuildScheduler(optimizer=optimizer, scheduler_cfg=scheduler_cfg)
        start_epoch, end_epoch = 1, scheduler_cfg['max_epochs']
        # load checkpoints
        if cmd_args.checkpointspath and os.path.exists(cmd_args.checkpointspath):
            checkpoints = loadcheckpoints(cmd_args.checkpointspath, logger_handle=logger_handle, cmd_args=cmd_args)
            try:
                segmentor.load_state_dict(checkpoints['model'])
            except Exception as e:
                logger_handle.warning(str(e) + '\n' + 'Try to load checkpoints by using strict=False')
                segmentor.load_state_dict(checkpoints['model'], strict=False)
            if 'optimizer' in checkpoints: 
                optimizer.load_state_dict(checkpoints['optimizer'])
            if 'cur_epoch' in checkpoints: 
                start_epoch = checkpoints['cur_epoch'] + 1
                scheduler.setstate({'cur_epoch': checkpoints['cur_epoch'], 'cur_iter': checkpoints['cur_iter']})
                assert checkpoints['cur_iter'] == len(dataloader) * checkpoints['cur_epoch']
            if 'amp' in checkpoints and fp16_type is not None:
                apex.amp.load_state_dict(checkpoints['amp'])
        else:
            cmd_args.checkpointspath = ''
        # parallel segmentor
        build_dist_model_cfg = self.cfg.SEGMENTOR_CFG.get('build_dist_model_cfg', {})
        build_dist_model_cfg.update({'device_ids': [cmd_args.local_rank]})
        segmentor = BuildDistributedModel(segmentor, build_dist_model_cfg)
        # print config
        if (cmd_args.local_rank == 0) and (int(os.environ.get('SLURM_PROCID', 0)) == 0):
            logger_handle.info(f'Config file path: {cfg_file_path}')
            logger_handle.info(f'DATASET_CFG: \n{cfg.DATASET_CFG}')
            logger_handle.info(f'DATALOADER_CFG: \n{cfg.DATALOADER_CFG}')
            logger_handle.info(f'OPTIMIZER_CFG: \n{cfg.OPTIMIZER_CFG}')
            logger_handle.info(f'SCHEDULER_CFG: \n{cfg.SCHEDULER_CFG}')
            logger_handle.info(f'LOSSES_CFG: \n{cfg.LOSSES_CFG}')
            logger_handle.info(f'SEGMENTOR_CFG: \n{cfg.SEGMENTOR_CFG}')
            logger_handle.info(f'INFERENCE_CFG: \n{cfg.INFERENCE_CFG}')
            logger_handle.info(f'COMMON_CFG: \n{cfg.COMMON_CFG}')
            logger_handle.info(f'Resume from: {cmd_args.checkpointspath}')
        # start to train the segmentor
        FloatTensor, losses_log_dict_memory = torch.cuda.FloatTensor, {}
        for epoch in range(start_epoch, end_epoch+1):
            # --set train
            segmentor.train()
            dataloader.sampler.set_epoch(epoch)
            # --train epoch
            for batch_idx, samples in enumerate(dataloader):
                learning_rate = scheduler.updatelr()
                images, targets = samples['image'].type(FloatTensor), {'segmentation': samples['segmentation'].type(FloatTensor), 'edge': samples['edge'].type(FloatTensor)}
                optimizer.zero_grad()
                if cfg.SEGMENTOR_CFG['type'] in ['memorynet', 'memorynetv2']:
                    loss, losses_log_dict = segmentor(images, targets, cfg.LOSSES_CFG, learning_rate=learning_rate, epoch=epoch)
                else:
                    loss, losses_log_dict = segmentor(images, targets, cfg.LOSSES_CFG)
                for key, value in losses_log_dict.items():
                    if key in losses_log_dict_memory: 
                        losses_log_dict_memory[key].append(value)
                    else: 
                        losses_log_dict_memory[key] = [value]
                if fp16_type is not None:
                    with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                scheduler.step()
                if (cmd_args.local_rank == 0) and (scheduler.cur_iter % cfg.COMMON_CFG['log_interval_iterations'] == 0) and (int(os.environ.get('SLURM_PROCID', 0)) == 0):
                    loss_log = ''
                    for key, value in losses_log_dict_memory.items():
                        loss_log += '%s %.4f, ' % (key, sum(value) / len(value))
                    losses_log_dict_memory = dict()
                    logger_handle.info(
                        f'[Epoch]: {epoch}/{end_epoch}, [Batch]: {batch_idx+1}/{len(dataloader)}, [Segmentor]: {cfg.SEGMENTOR_CFG["type"]}-{cfg.SEGMENTOR_CFG["backbone"]["type"]}, '
                        f'[DATASET]: {cfg.DATASET_CFG["type"]}, [LEARNING_RATE]: {learning_rate}\n\t[LOSS]: {loss_log}'
                    )
            scheduler.cur_epoch = epoch
            # --save checkpoints
            if (epoch % cfg.COMMON_CFG['save_interval_epochs'] == 0) or (epoch == end_epoch):
                state_dict = scheduler.state()
                state_dict['model'] = segmentor.module.state_dict()
                if fp16_type is not None:
                    state_dict['amp'] = apex.amp.state_dict()
                savepath = os.path.join(cfg.COMMON_CFG['work_dir'], 'epoch_%s.pth' % epoch)
                if (cmd_args.local_rank == 0) and (int(os.environ.get('SLURM_PROCID', 0)) == 0):
                    savecheckpoints(state_dict, savepath, logger_handle, cmd_args=cmd_args)
            # --eval checkpoints
            if (epoch % cfg.COMMON_CFG['eval_interval_epochs'] == 0) or (epoch == end_epoch):
                self.evaluate(segmentor)
    '''evaluate'''
    def evaluate(self, segmentor):
        cfg, ngpus_per_node, cmd_args, logger_handle = self.cfg, self.ngpus_per_node, self.cmd_args, self.logger_handle
        # TODO: bug occurs if use --pyt bash
        rank_id = int(os.environ['SLURM_PROCID']) if 'SLURM_PROCID' in os.environ else cmd_args.local_rank
        # build dataset and dataloader
        dataset = BuildDataset(mode='TEST', logger_handle=logger_handle, dataset_cfg=copy.deepcopy(cfg.DATASET_CFG))
        dataloader_cfg = copy.deepcopy(cfg.DATALOADER_CFG)
        batch_size, num_workers = ngpus_per_node, dataloader_cfg['test']['num_workers']
        batch_size_per_node = batch_size // ngpus_per_node
        num_workers_per_node = num_workers // ngpus_per_node
        dataloader_cfg['test'].update({'batch_size': batch_size_per_node, 'num_workers': num_workers_per_node})
        dataloader = BuildDistributedDataloader(dataset=dataset, dataloader_cfg=dataloader_cfg['test'])
        # start to eval
        segmentor.eval()
        segmentor.module.mode = 'TEST'
        inference_cfg, all_preds, all_gts = cfg.INFERENCE_CFG, [], []
        align_corners = segmentor.module.align_corners
        FloatTensor = torch.cuda.FloatTensor
        use_probs_before_resize = inference_cfg['tricks']['use_probs_before_resize']
        assert inference_cfg['mode'] in ['whole', 'slide']
        with torch.no_grad():
            dataloader.sampler.set_epoch(0)
            pbar = tqdm(enumerate(dataloader))
            for batch_idx, samples in pbar:
                pbar.set_description('Processing %s/%s in rank %s' % (batch_idx+1, len(dataloader), rank_id))
                imageids, images, widths, heights, gts = samples['id'], samples['image'].type(FloatTensor), samples['width'], samples['height'], samples['groundtruth']
                if inference_cfg['mode'] == 'whole':
                    outputs = segmentor(images)
                    if use_probs_before_resize:
                        outputs = F.softmax(outputs, dim=1)
                else:
                    opts = inference_cfg['opts']
                    stride_h, stride_w = opts['stride']
                    cropsize_h, cropsize_w = opts['cropsize']
                    batch_size, _, image_h, image_w = images.size()
                    num_grids_h = max(image_h - cropsize_h + stride_h - 1, 0) // stride_h + 1
                    num_grids_w = max(image_w - cropsize_w + stride_w - 1, 0) // stride_w + 1
                    outputs = images.new_zeros((batch_size, cfg.SEGMENTOR_CFG['num_classes'], image_h, image_w))
                    count_mat = images.new_zeros((batch_size, 1, image_h, image_w))
                    for h_idx in range(num_grids_h):
                        for w_idx in range(num_grids_w):
                            x1, y1 = w_idx * stride_w, h_idx * stride_h
                            x2, y2 = min(x1 + cropsize_w, image_w), min(y1 + cropsize_h, image_h)
                            x1, y1 = max(x2 - cropsize_w, 0), max(y2 - cropsize_h, 0)
                            crop_images = images[:, :, y1:y2, x1:x2]
                            outputs_crop = segmentor(crop_images)
                            outputs_crop = F.interpolate(outputs_crop, size=crop_images.size()[2:], mode='bilinear', align_corners=align_corners)
                            if use_probs_before_resize: 
                                outputs_crop = F.softmax(outputs_crop, dim=1)
                            outputs += F.pad(outputs_crop, (int(x1), int(outputs.shape[3] - x2), int(y1), int(outputs.shape[2] - y2)))
                            count_mat[:, :, y1:y2, x1:x2] += 1
                    assert (count_mat == 0).sum() == 0
                    outputs = outputs / count_mat
                for idx in range(len(outputs)):
                    output = F.interpolate(outputs[idx: idx+1], size=(heights[idx], widths[idx]), mode='bilinear', align_corners=align_corners)
                    pred = (torch.argmax(output[0], dim=0)).cpu().numpy().astype(np.int32)
                    all_preds.append([imageids[idx], pred])
                    gt = gts[idx].cpu().numpy().astype(np.int32)
                    gt[gt >= dataset.num_classes] = -1
                    all_gts.append(gt)
        # collect eval results and calculate the metric
        filename = cfg.COMMON_CFG['resultsavepath'].split('/')[-1].split('.')[0] + f'_{rank_id}.' + cfg.COMMON_CFG['resultsavepath'].split('.')[-1]
        with open(os.path.join(cfg.COMMON_CFG['work_dir'], filename), 'wb') as fp:
            pickle.dump([all_preds, all_gts], fp)
        rank = torch.tensor([rank_id], device='cuda')
        rank_list = [rank.clone() for _ in range(ngpus_per_node)]
        dist.all_gather(rank_list, rank)
        logger_handle.info('Rank %s finished' % int(rank.item()))
        if rank_id == 0:
            all_preds_gather, all_gts_gather = [], []
            for rank in rank_list:
                rank = str(int(rank.item()))
                filename = cfg.COMMON_CFG['resultsavepath'].split('/')[-1].split('.')[0] + f'_{rank}.' + cfg.COMMON_CFG['resultsavepath'].split('.')[-1]
                fp = open(os.path.join(cfg.COMMON_CFG['work_dir'], filename), 'rb')
                all_preds, all_gts = pickle.load(fp)
                all_preds_gather += all_preds
                all_gts_gather += all_gts
            all_preds, all_gts = all_preds_gather, all_gts_gather
            all_preds_filtered, all_gts_filtered, all_ids = [], [], []
            for idx, pred in enumerate(all_preds):
                if pred[0] in all_ids: 
                    continue
                all_ids.append(pred[0])
                all_preds_filtered.append(pred[1])
                all_gts_filtered.append(all_gts[idx])
            all_preds, all_gts = all_preds_filtered, all_gts_filtered
            logger_handle.info('All Finished, all_preds: %s, all_gts: %s' % (len(all_preds), len(all_gts)))
            result = dataset.evaluate(
                predictions=all_preds, 
                groundtruths=all_gts, 
                metric_list=cfg.INFERENCE_CFG.get('metric_list', ['iou', 'miou']),
                num_classes=cfg.SEGMENTOR_CFG['num_classes'],
                ignore_index=-1,
            )
            logger_handle.info(result)
        segmentor.train()
        segmentor.module.mode = 'TRAIN'


'''main'''
def main():
    # parse arguments
    args = parseArgs()
    cfg, cfg_file_path = BuildConfig(args.cfgfilepath)
    # check work dir
    checkdir(cfg.COMMON_CFG['work_dir'])
    # initialize logger_handle
    logger_handle = Logger(cfg.COMMON_CFG['logfilepath'])
    # number of gpus, for distribued training, only support a process for a GPU
    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node != args.nproc_per_node:
        if (args.local_rank == 0) and (int(os.environ.get('SLURM_PROCID', 0)) == 0): 
            logger_handle.warning('ngpus_per_node is not equal to nproc_per_node, force ngpus_per_node = nproc_per_node by default')
        ngpus_per_node = args.nproc_per_node
    # instanced Trainer
    client = Trainer(cfg=cfg, ngpus_per_node=ngpus_per_node, logger_handle=logger_handle, cmd_args=args, cfg_file_path=cfg_file_path)
    client.start()


'''debug'''
if __name__ == '__main__':
    main()