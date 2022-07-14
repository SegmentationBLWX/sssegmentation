'''
Function:
    Train the segmentor
Author:
    Zhenchao Jin
'''
import os
import copy
import torch
import warnings
import argparse
import torch.nn as nn
import torch.distributed as dist
from configs import BuildConfig
from modules import (
    BuildDataset, BuildDistributedDataloader, BuildDistributedModel, BuildOptimizer, adjustLearningRate, clipGradients, initslurm,
    BuildLoss, BuildBackbone, BuildSegmentor, BuildPixelSampler, Logger, setRandomSeed, BuildPalette, checkdir, loadcheckpoints, savecheckpoints
)
warnings.filterwarnings('ignore')


'''parse arguments in command line'''
def parseArgs():
    parser = argparse.ArgumentParser(description='SSSegmentation is an open source strongly supervised semantic segmentation toolbox based on PyTorch')
    parser.add_argument('--local_rank', dest='local_rank', help='node rank for distributed training', default=0, type=int)
    parser.add_argument('--nproc_per_node', dest='nproc_per_node', help='number of process per node', default=8, type=int)
    parser.add_argument('--cfgfilepath', dest='cfgfilepath', help='config file path you want to use', type=str, required=True)
    parser.add_argument('--checkpointspath', dest='checkpointspath', help='checkpoints you want to resume from', default='', type=str)
    args = parser.parse_args()
    initslurm(args, '29000')
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
        learning_rate = optimizer_cfg[optimizer_cfg['type']]['learning_rate']
        optimizer = BuildOptimizer(segmentor, optimizer_cfg)
        start_epoch, end_epoch = 1, optimizer_cfg['max_epochs']
        # build fp16
        fp16_cfg = self.cfg.SEGMENTOR_CFG.get('fp16', {'is_on': False, 'opts': {'opt_level': 'O1'}})
        if fp16_cfg['is_on']:
            import apex
            segmentor, optimizer = apex.amp.initialize(segmentor, optimizer, **fp16_cfg['opts'])
            for m in segmentor.modules():
                if hasattr(m, 'fp16_enabled'):
                    m.fp16_enabled = True
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
            if 'epoch' in checkpoints: 
                start_epoch = checkpoints['epoch'] + 1
        else:
            cmd_args.checkpointspath = ''
        num_iters, max_iters = (start_epoch - 1) * len(dataloader), end_epoch * len(dataloader)
        # parallel segmentor
        segmentor = BuildDistributedModel(segmentor, {'device_ids': [cmd_args.local_rank]})
        # print config
        if cmd_args.local_rank == 0:
            logger_handle.info(f'Config file path: {cfg_file_path}')
            logger_handle.info(f'DATASET_CFG: \n{cfg.DATASET_CFG}')
            logger_handle.info(f'DATALOADER_CFG: \n{cfg.DATALOADER_CFG}')
            logger_handle.info(f'OPTIMIZER_CFG: \n{cfg.OPTIMIZER_CFG}')
            logger_handle.info(f'LOSSES_CFG: \n{cfg.LOSSES_CFG}')
            logger_handle.info(f'SEGMENTOR_CFG: \n{cfg.SEGMENTOR_CFG}')
            logger_handle.info(f'INFERENCE_CFG: \n{cfg.INFERENCE_CFG}')
            logger_handle.info(f'COMMON_CFG: \n{cfg.COMMON_CFG}')
            logger_handle.info(f'Resume from: {cmd_args.checkpointspath}')
        # start to train the segmentor
        FloatTensor = torch.cuda.FloatTensor
        losses_log_dict_memory = {}
        for epoch in range(start_epoch, end_epoch+1):
            # --set train
            segmentor.train()
            dataloader.sampler.set_epoch(epoch)
            # --adjust lr if necessary
            if optimizer_cfg['adjust_period'] == 'epoch':
                optimizer_cfg['policy']['opts'].update({'num_iters': num_iters, 'max_iters': max_iters, 'num_epochs': epoch})
                learning_rate = adjustLearningRate(optimizer, copy.deepcopy(optimizer_cfg))
            # --train epoch
            for batch_idx, samples in enumerate(dataloader):
                if optimizer_cfg['adjust_period'] == 'iteration':
                    optimizer_cfg['policy']['opts'].update({'num_iters': num_iters, 'max_iters': max_iters, 'num_epochs': epoch})
                    learning_rate = adjustLearningRate(optimizer, optimizer_cfg)
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
                if fp16_cfg['is_on']:
                    with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optimizer.step()
                num_iters += 1
                if (cmd_args.local_rank == 0) and (num_iters % cfg.COMMON_CFG['log_interval_iterations'] == 0):
                    loss_log = ''
                    for key, value in losses_log_dict_memory.items():
                        loss_log += '%s %.4f, ' % (key, sum(value) / len(value))
                    losses_log_dict_memory = dict()
                    logger_handle.info(
                        f'[Epoch]: {epoch}/{end_epoch}, [Batch]: {batch_idx+1}/{len(dataloader)}, [Segmentor]: {cfg.SEGMENTOR_CFG["type"]}-{cfg.SEGMENTOR_CFG["backbone"]["type"]}, '
                        f'[DATASET]: {cfg.DATASET_CFG["type"]}, [LEARNING_RATE]: {learning_rate}\n\t[LOSS]: {loss_log}'
                    )
            # --save checkpoints
            if (epoch % cfg.COMMON_CFG['save_interval_epochs'] == 0) or (epoch == end_epoch):
                state_dict = {
                    'epoch': epoch,
                    'model': segmentor.module.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                savepath = os.path.join(cfg.COMMON_CFG['work_dir'], 'epoch_%s.pth' % epoch)
                if cmd_args.local_rank == 0: savecheckpoints(state_dict, savepath, logger_handle, cmd_args=cmd_args)
            # --eval checkpoints
            if (epoch % cfg.COMMON_CFG['eval_interval_epochs'] == 0) or (epoch == end_epoch):
                pass


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
        if args.local_rank == 0: 
            logger_handle.warning('ngpus_per_node is not equal to nproc_per_node, force ngpus_per_node = nproc_per_node by default')
        ngpus_per_node = args.nproc_per_node
    # instanced Trainer
    client = Trainer(cfg=cfg, ngpus_per_node=ngpus_per_node, logger_handle=logger_handle, cmd_args=args, cfg_file_path=cfg_file_path)
    client.start()


'''debug'''
if __name__ == '__main__':
    main()