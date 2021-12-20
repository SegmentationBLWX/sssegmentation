'''
Function:
    train the segmentor
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
from modules import *
from cfgs import BuildConfig
warnings.filterwarnings('ignore')


'''parse arguments in command line'''
def parseArgs():
    parser = argparse.ArgumentParser(description='SSSegmentation is an open source strongly supervised semantic segmentation toolbox based on PyTorch')
    parser.add_argument('--local_rank', dest='local_rank', help='node rank for distributed training', default=0, type=int)
    parser.add_argument('--nproc_per_node', dest='nproc_per_node', help='number of process per node', default=4, type=int)
    parser.add_argument('--cfgfilepath', dest='cfgfilepath', help='config file path you want to use', type=str, required=True)
    parser.add_argument('--checkpointspath', dest='checkpointspath', help='checkpoints you want to resume from', default='', type=str)
    args = parser.parse_args()
    return args


'''Trainer'''
class Trainer():
    def __init__(self, **kwargs):
        # set attribute
        for key, value in kwargs.items(): setattr(self, key, value)
        self.use_cuda = torch.cuda.is_available()
        # modify config for consistency
        if not self.use_cuda:
            if self.cmd_args.local_rank == 0: self.logger_handle.warning('Cuda is not available, only cpu is used to train the segmentor...')
            self.cfg.SEGMENTOR_CFG['distributed']['is_on'] = False
            self.cfg.DATALOADER_CFG['train']['type'] = 'nondistributed'
        if self.cfg.SEGMENTOR_CFG['distributed']['is_on']:
            self.cfg.SEGMENTOR_CFG['is_multi_gpus'] = True
            self.cfg.DATALOADER_CFG['train']['type'] = 'distributed'
        # init distributed training if necessary
        distributed_cfg = self.cfg.SEGMENTOR_CFG['distributed']
        if distributed_cfg['is_on']:
            dist.init_process_group(backend=distributed_cfg.get('backend', 'nccl'))
    '''start trainer'''
    def start(self):
        cfg, logger_handle, use_cuda, cmd_args, cfg_file_path = self.cfg, self.logger_handle, self.use_cuda, self.cmd_args, self.cfg_file_path
        distributed_cfg, common_cfg = self.cfg.SEGMENTOR_CFG['distributed'], self.cfg.COMMON_CFG['train']
        # instanced dataset and dataloader
        dataset = BuildDataset(mode='TRAIN', logger_handle=logger_handle, dataset_cfg=copy.deepcopy(cfg.DATASET_CFG))
        assert dataset.num_classes == cfg.SEGMENTOR_CFG['num_classes'], 'parsed config file %s error...' % cfg_file_path
        dataloader_cfg = copy.deepcopy(cfg.DATALOADER_CFG)
        if distributed_cfg['is_on']:
            batch_size, num_workers = dataloader_cfg['train']['batch_size'], dataloader_cfg['train']['num_workers']
            batch_size //= self.ngpus_per_node
            num_workers //= self.ngpus_per_node
            assert batch_size * self.ngpus_per_node == dataloader_cfg['train']['batch_size'], 'unsuitable batch_size...'
            assert num_workers * self.ngpus_per_node == dataloader_cfg['train']['num_workers'], 'unsuitable num_workers...'
            dataloader_cfg['train'].update({'batch_size': batch_size, 'num_workers': num_workers})
        dataloader = BuildParallelDataloader(mode='TRAIN', dataset=dataset, cfg=copy.deepcopy(dataloader_cfg))
        # instanced segmentor
        segmentor = BuildSegmentor(segmentor_cfg=copy.deepcopy(cfg.SEGMENTOR_CFG), mode='TRAIN')
        if distributed_cfg['is_on']:
            torch.cuda.set_device(cmd_args.local_rank)
            segmentor.cuda(cmd_args.local_rank)
        else:
            if use_cuda: segmentor = segmentor.cuda()
        torch.backends.cudnn.benchmark = cfg.SEGMENTOR_CFG['benchmark']
        # build optimizer
        optimizer_cfg = cfg.OPTIMIZER_CFG
        learning_rate = optimizer_cfg[optimizer_cfg['type']]['learning_rate']
        optimizer = BuildOptimizer(segmentor, copy.deepcopy(optimizer_cfg))
        start_epoch = 1
        end_epoch = cfg.OPTIMIZER_CFG['max_epochs']
        # whether use fp16
        fp16_cfg = self.cfg.SEGMENTOR_CFG.get('fp16', {'is_on': False, 'opts': {'opt_level': 'O1'}})
        if fp16_cfg['is_on']:
            import apex
            segmentor, optimizer = apex.amp.initialize(segmentor, optimizer, **fp16_cfg['opts'])
            for m in segmentor.modules():
                if hasattr(m, "fp16_enabled"):
                    m.fp16_enabled = True
        # load checkpoints
        if cmd_args.checkpointspath and os.path.exists(cmd_args.checkpointspath):
            checkpoints = loadcheckpoints(cmd_args.checkpointspath, logger_handle=logger_handle, cmd_args=cmd_args)
            try:
                segmentor.load_state_dict(checkpoints['model'])
            except Exception as e:
                logger_handle.warning(str(e) + '\n' + 'Try to load checkpoints by using strict=False...')
                segmentor.load_state_dict(checkpoints['model'], strict=False)
            if 'optimizer' in checkpoints: optimizer.load_state_dict(checkpoints['optimizer'])
            if 'epoch' in checkpoints: start_epoch = checkpoints['epoch'] + 1
        else:
            cmd_args.checkpointspath = ''
        num_iters, max_iters = (start_epoch - 1) * len(dataloader), end_epoch * len(dataloader)
        # parallel
        if use_cuda and cfg.SEGMENTOR_CFG['is_multi_gpus']:
            is_distributed_on = cfg.SEGMENTOR_CFG['distributed']['is_on']
            segmentor = BuildParallelModel(segmentor, is_distributed_on, device_ids=[cmd_args.local_rank] if is_distributed_on else None)
        # print information
        if cmd_args.local_rank == 0:
            logger_handle.info('Dataset: %s-%s, Number of images: %s' % (cfg.DATASET_CFG['type'], cfg.DATASET_CFG['train']['set'], len(dataset)))
            logger_handle.info('Segmentor: %s, Backbone: %s-%s' % (cfg.SEGMENTOR_CFG['type'], cfg.SEGMENTOR_CFG['backbone']['series'], cfg.SEGMENTOR_CFG['backbone']['type']))
            logger_handle.info('Resume from: %s' % cmd_args.checkpointspath)
            logger_handle.info('Config file path: %s' % cfg_file_path)
        # start to train the segmentor
        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        losses_log_dict_memory = {}
        for epoch in range(start_epoch, end_epoch+1):
            # --set train
            segmentor.train()
            if cfg.SEGMENTOR_CFG['distributed']['is_on']: dataloader.sampler.set_epoch(epoch)
            # --adjust lr if necessary
            if optimizer_cfg['adjust_period'] == 'epoch':
                optimizer_cfg['policy']['opts'].update({'num_iters': num_iters, 'max_iters': max_iters, 'num_epochs': epoch})
                learning_rate = adjustLearningRate(optimizer, copy.deepcopy(optimizer_cfg))
            # --log information
            if cmd_args.local_rank == 0: logger_handle.info('Start epoch %s...' % epoch)
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
                if not distributed_cfg['is_on']:
                    loss = loss.mean()
                    for key, value in losses_log_dict.items():
                        losses_log_dict[key] = value.mean().item()
                for key, value in losses_log_dict.items():
                    if key in losses_log_dict_memory: losses_log_dict_memory[key].append(value)
                    else: losses_log_dict_memory[key] = [value]
                if fp16_cfg['is_on']:
                    with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optimizer.step()
                num_iters += 1
                if (cmd_args.local_rank == 0) and (num_iters % common_cfg['loginterval'] == 0):
                    loss_log = ''
                    for key, value in losses_log_dict_memory.items():
                        loss_log += '%s %.4f, ' % (key, sum(value) / len(value))
                    losses_log_dict_memory = dict()
                    logger_handle.info('[EPOCH]: %s/%s, [BATCH]: %s/%s, [LEARNING_RATE]: %s, [DATASET]: %s\n\t[LOSS]: %s' % \
                                        (epoch, end_epoch, (batch_idx+1), len(dataloader), learning_rate, cfg.DATASET_CFG['type'], loss_log))
            # --save checkpoints
            if (epoch % common_cfg['saveinterval'] == 0) or (epoch == end_epoch):
                state_dict = {
                    'epoch': epoch,
                    'model': segmentor.module.state_dict() if cfg.SEGMENTOR_CFG['is_multi_gpus'] else segmentor.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                savepath = os.path.join(common_cfg['backupdir'], 'epoch_%s.pth' % epoch)
                if cmd_args.local_rank == 0: savecheckpoints(state_dict, savepath, logger_handle, cmd_args=cmd_args)


'''main'''
def main():
    # parse arguments
    args = parseArgs()
    cfg, cfg_file_path = BuildConfig(args.cfgfilepath)
    # check backup dir
    common_cfg = cfg.COMMON_CFG['train']
    checkdir(common_cfg['backupdir'])
    # initialize logger_handle
    logger_handle = Logger(common_cfg['logfilepath'])
    # number of gpus, for distribued training, only support a process for a GPU
    ngpus_per_node = torch.cuda.device_count()
    if (ngpus_per_node != args.nproc_per_node) and cfg.SEGMENTOR_CFG['distributed']['is_on']:
        if args.local_rank == 0: logger_handle.warning('ngpus_per_node is not equal to nproc_per_node, force ngpus_per_node = nproc_per_node by default...')
        ngpus_per_node = args.nproc_per_node
    # instanced Trainer
    client = Trainer(cfg=cfg, ngpus_per_node=ngpus_per_node, logger_handle=logger_handle, cmd_args=args, cfg_file_path=cfg_file_path)
    client.start()


'''debug'''
if __name__ == '__main__':
    main()