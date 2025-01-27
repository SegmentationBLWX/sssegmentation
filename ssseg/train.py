'''
Function:
    Implementation of Trainer
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
from datetime import timedelta
try:
    from modules import (
        BuildDistributedDataloader, BuildDistributedModel, touchdirs, loadckpts, saveckpts, judgefileexist, postprocesspredgtpairs, initslurm, ismainprocess,
        BuildDataset,  BuildSegmentor, BuildOptimizer, BuildScheduler, BuildLoggerHandle, TrainingLoggingManager, ConfigParser, EMASegmentor, SSSegInputStructure
    )
except:
    from .modules import (
        BuildDistributedDataloader, BuildDistributedModel, touchdirs, loadckpts, saveckpts, judgefileexist, postprocesspredgtpairs, initslurm, ismainprocess,
        BuildDataset,  BuildSegmentor, BuildOptimizer, BuildScheduler, BuildLoggerHandle, TrainingLoggingManager, ConfigParser, EMASegmentor, SSSegInputStructure
    )
warnings.filterwarnings('ignore')


'''parsecmdargs'''
def parsecmdargs():
    parser = argparse.ArgumentParser(description='SSSegmentation is an open source supervised semantic segmentation toolbox based on PyTorch.')
    parser.add_argument('--local_rank', '--local-rank', dest='local_rank', help='The rank of the worker within a local worker group.', default=0, type=int)
    parser.add_argument('--nproc_per_node', dest='nproc_per_node', help='The number of processes per node.', default=8, type=int)
    parser.add_argument('--cfgfilepath', dest='cfgfilepath', help='The config file path which is used to customize segmentors.', type=str, required=True)
    parser.add_argument('--ckptspath', dest='ckptspath', help='Specify the checkpoint from which to resume training.', default='', type=str)
    parser.add_argument('--slurm', dest='slurm', help='Please add --slurm if you are using slurm to spawn training jobs.', default=False, action='store_true')
    cmd_args = parser.parse_args()
    if torch.__version__.startswith('2.'):
        cmd_args.local_rank = int(os.environ['LOCAL_RANK'])
    if cmd_args.slurm:
        initslurm(cmd_args, str(8888 + random.randint(0, 1000)))
    else:
        if 'RANK' not in os.environ:
            os.environ['RANK'] = str(cmd_args.local_rank)
        if 'LOCAL_RANK' not in os.environ:
            os.environ['LOCAL_RANK'] = str(cmd_args.local_rank)
        if 'WORLD_SIZE' not in os.environ:
            os.environ['WORLD_SIZE'] = str(cmd_args.nproc_per_node)
        if 'LOCAL_WORLD_SIZE' not in os.environ:
            os.environ['LOCAL_WORLD_SIZE'] = str(cmd_args.nproc_per_node)
    return cmd_args


'''Trainer'''
class Trainer():
    def __init__(self, cmd_args):
        # parse config file
        config_parser = ConfigParser()
        cfg, cfg_file_path = config_parser(cmd_args.cfgfilepath)
        # touch work dir
        touchdirs(cfg.SEGMENTOR_CFG['work_dir'])
        config_parser.save(cfg.SEGMENTOR_CFG['work_dir'])
        # initialize logger_handle and training_logging_manager
        logger_handle = BuildLoggerHandle(cfg.SEGMENTOR_CFG['logger_handle_cfg'])
        training_logging_manager_cfg = copy.deepcopy(cfg.SEGMENTOR_CFG['training_logging_manager_cfg'])
        if ('logger_handle_cfg' not in training_logging_manager_cfg) and ('logger_handle' not in training_logging_manager_cfg):
            training_logging_manager_cfg['logger_handle'] = logger_handle
        training_logging_manager = TrainingLoggingManager(**training_logging_manager_cfg)
        # set attributes
        self.cfg = cfg
        self.num_total_processes = int(os.environ['WORLD_SIZE'])
        self.logger_handle = logger_handle
        self.cmd_args = cmd_args
        self.cfg_file_path = cfg_file_path
        self.training_logging_manager = training_logging_manager
        assert torch.cuda.is_available(), 'cuda is not available'
        # init distributed training
        dist.init_process_group(**self.cfg.SEGMENTOR_CFG.get('init_process_group_cfg', {'backend': 'nccl', 'timeout': timedelta(seconds=36000)}))
        # open full fp32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    '''start trainer'''
    def start(self):
        # initialize necessary variables
        cfg, num_total_processes, logger_handle, cmd_args, cfg_file_path, training_logging_manager = self.cfg, self.num_total_processes, self.logger_handle, self.cmd_args, self.cfg_file_path, self.training_logging_manager
        # build dataset and dataloader
        dataset = BuildDataset(mode='TRAIN', logger_handle=logger_handle, dataset_cfg=cfg.SEGMENTOR_CFG['dataset'])
        assert dataset.num_classes == cfg.SEGMENTOR_CFG['num_classes'], 'parsed config file %s error' % cfg_file_path
        dataloader_cfg = copy.deepcopy(cfg.SEGMENTOR_CFG['dataloader'])
        auto_adapt_to_expected_train_bs = dataloader_cfg.pop('auto_adapt_to_expected_train_bs')
        expected_total_train_bs_for_assert = dataloader_cfg.pop('expected_total_train_bs_for_assert')
        if auto_adapt_to_expected_train_bs:
            dataloader_cfg['train']['batch_size_per_gpu'] = expected_total_train_bs_for_assert // num_total_processes
        dataloader_cfg['train']['batch_size'], dataloader_cfg['train']['num_workers'] = dataloader_cfg['train'].pop('batch_size_per_gpu'), dataloader_cfg['train'].pop('num_workers_per_gpu')
        dataloader_cfg['test']['batch_size'], dataloader_cfg['test']['num_workers'] = dataloader_cfg['test'].pop('batch_size_per_gpu'), dataloader_cfg['test'].pop('num_workers_per_gpu')
        assert expected_total_train_bs_for_assert == dataloader_cfg['train']['batch_size'] * num_total_processes
        dataloader = BuildDistributedDataloader(dataset=dataset, dataloader_cfg=dataloader_cfg['train'])
        # build segmentor
        segmentor = BuildSegmentor(segmentor_cfg=cfg.SEGMENTOR_CFG, mode='TRAIN')
        dist.barrier()
        torch.cuda.set_device(cmd_args.local_rank)
        segmentor.cuda(cmd_args.local_rank)
        torch.backends.cudnn.benchmark = cfg.SEGMENTOR_CFG['benchmark']
        # build optimizer
        optimizer = BuildOptimizer(segmentor, cfg.SEGMENTOR_CFG['scheduler']['optimizer'])
        # build fp16
        fp16_cfg = copy.deepcopy(self.cfg.SEGMENTOR_CFG.get('fp16_cfg', {'type': None}))
        fp16_type, grad_scaler = fp16_cfg.pop('type'), None
        assert fp16_type in [None, 'apex', 'pytorch']
        if fp16_type in ['apex']:
            import apex
            segmentor, optimizer = apex.amp.initialize(segmentor, optimizer, **fp16_cfg['initialize'])
        elif fp16_type in ['pytorch']:
            from torch.cuda.amp import autocast
            from torch.cuda.amp import GradScaler
            from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks
            grad_scaler = GradScaler(**fp16_cfg['grad_scaler'])
        # build ema
        ema_cfg = copy.deepcopy(self.cfg.SEGMENTOR_CFG.get('ema_cfg', {'momentum': None, 'device': 'cpu'}))
        if ema_cfg['momentum'] is not None:
            segmentor_ema = EMASegmentor(segmentor=segmentor, **ema_cfg)
        # build scheduler
        scheduler_cfg = copy.deepcopy(cfg.SEGMENTOR_CFG['scheduler'])
        scheduler_cfg.update({
            'lr': cfg.SEGMENTOR_CFG['scheduler']['optimizer']['lr'],
            'iters_per_epoch': len(dataloader),
            'params_rules': cfg.SEGMENTOR_CFG['scheduler']['optimizer']['params_rules'],
        })
        scheduler = BuildScheduler(optimizer=optimizer, scheduler_cfg=scheduler_cfg)
        start_epoch, end_epoch = 1, scheduler_cfg['max_epochs']
        # load ckpts
        if cmd_args.ckptspath and judgefileexist(cmd_args.ckptspath):
            ckpts = loadckpts(cmd_args.ckptspath)
            try:
                segmentor.load_state_dict(ckpts['model'])
            except Exception as e:
                logger_handle.warning(str(e) + '\n' + 'Try to load ckpts by using strict=False', main_process_only=True)
                segmentor.load_state_dict(ckpts['model'], strict=False)
            if 'optimizer' in ckpts: 
                optimizer.load_state_dict(ckpts['optimizer'])
            if 'cur_epoch' in ckpts: 
                start_epoch = ckpts['cur_epoch'] + 1
                scheduler.setstate({'cur_epoch': ckpts['cur_epoch'], 'cur_iter': ckpts['cur_iter']})
                assert ckpts['cur_iter'] == len(dataloader) * ckpts['cur_epoch']
            if 'amp' in ckpts and fp16_type in ['apex']:
                apex.amp.load_state_dict(ckpts['amp'])
            if 'grad_scaler' in ckpts and fp16_type in ['pytorch']:
                grad_scaler.load_state_dict(ckpts['grad_scaler'])
            if 'model_ema' in ckpts and ema_cfg['momentum'] is not None:
                segmentor_ema.setstate(ckpts['model_ema'], strict=True)
        else:
            cmd_args.ckptspath = ''
        # parallel segmentor
        build_dist_model_cfg = copy.deepcopy(self.cfg.SEGMENTOR_CFG.get('build_dist_model_cfg', {}))
        build_dist_model_cfg.update({'device_ids': [cmd_args.local_rank]})
        segmentor = BuildDistributedModel(segmentor, build_dist_model_cfg)
        if fp16_type in ['pytorch']:
            segmentor.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
        # print config
        logger_handle.info(f'Config file path: {cfg_file_path}', main_process_only=True)
        logger_handle.info(f'Config details: \n{cfg.SEGMENTOR_CFG}', main_process_only=True)
        logger_handle.info(f'Resume from: {cmd_args.ckptspath}', main_process_only=True)
        # start to train the segmentor
        for epoch in range(start_epoch, end_epoch+1):
            # --set train
            segmentor.train()
            dataloader.sampler.set_epoch(epoch)
            # --train epoch
            for batch_idx, samples_meta in enumerate(dataloader):
                learning_rate = scheduler.updatelr()
                data_meta = SSSegInputStructure(
                    mode='TRAIN', images=samples_meta['image'].type(torch.cuda.FloatTensor), seg_targets=samples_meta['seg_target'].type(torch.cuda.LongTensor),
                    edge_targets=samples_meta['edge_target'].type(torch.cuda.LongTensor) if 'edge_target' in samples_meta else None,
                    img2aug_pos_mapper=samples_meta['img2aug_pos_mapper'].type(torch.cuda.LongTensor) if 'img2aug_pos_mapper' in samples_meta else None,
                )
                optimizer.zero_grad()
                forward_kwargs = {'learning_rate': learning_rate, 'epoch': epoch} if cfg.SEGMENTOR_CFG['type'] in ['MCIBI', 'MCIBIPlusPlus'] else {}
                if fp16_type in ['pytorch']:
                    with autocast(**fp16_cfg['autocast']):
                        loss, losses_log_dict = segmentor(data_meta, **forward_kwargs).getsetvariables().values()
                else:
                    loss, losses_log_dict = segmentor(data_meta, **forward_kwargs).getsetvariables().values()
                if fp16_type in ['apex']:
                    with apex.amp.scale_loss(loss, optimizer, **fp16_cfg['scale_loss']) as scaled_loss:
                        scaled_loss.backward()
                elif fp16_type in ['pytorch']:
                    grad_scaler.scale(loss).backward()
                else:
                    loss.backward()
                scheduler.step(grad_scaler=grad_scaler)
                if ema_cfg['momentum'] is not None:
                    segmentor_ema.update(segmentor=segmentor)
                basic_log_dict = {
                    'cur_epoch': epoch, 'max_epochs': end_epoch, 'cur_iter': scheduler.cur_iter, 'max_iters': scheduler.max_iters,
                    'cur_iter_in_cur_epoch': batch_idx+1, 'max_iters_in_cur_epoch': len(dataloader), 'segmentor': cfg.SEGMENTOR_CFG['type'], 
                    'backbone': cfg.SEGMENTOR_CFG['backbone']['structure_type'], 'dataset': cfg.SEGMENTOR_CFG['dataset']['type'], 
                    'learning_rate': learning_rate, 'ema': ema_cfg['momentum'] is not None, 'fp16': fp16_type is not None,
                }
                training_logging_manager.update(basic_log_dict, losses_log_dict)
                training_logging_manager.autolog(main_process_only=True)
            scheduler.cur_epoch = epoch
            # --save ckpts
            if (epoch % cfg.SEGMENTOR_CFG['save_interval_epochs'] == 0) or (epoch == end_epoch):
                state_dict = scheduler.state()
                state_dict['model'] = segmentor.module.state_dict()
                if ema_cfg['momentum'] is not None:
                    state_dict['model_ema'] = segmentor_ema.state()
                if fp16_type in ['apex']:
                    state_dict['amp'] = apex.amp.state_dict()
                elif fp16_type in ['pytorch']:
                    state_dict['grad_scaler'] = grad_scaler.state_dict()
                savepath = os.path.join(cfg.SEGMENTOR_CFG['work_dir'], f'checkpoints-epoch-{epoch}.pth')
                if ismainprocess():
                    saveckpts(state_dict, savepath)
            # --eval ckpts
            if (epoch % cfg.SEGMENTOR_CFG['eval_interval_epochs'] == 0) or (epoch == end_epoch):
                self.logger_handle.info(f'Evaluate {cfg.SEGMENTOR_CFG["type"]} at epoch {epoch}', main_process_only=True)
                self.evaluate(segmentor)
                if ema_cfg['momentum'] is not None:
                    self.logger_handle.info(f'Evaluate EMA of {cfg.SEGMENTOR_CFG["type"]} at epoch {epoch}', main_process_only=True)
                    self.evaluate(segmentor_ema.segmentor_ema.cuda(cmd_args.local_rank))
    '''evaluate'''
    def evaluate(self, segmentor):
        cfg, logger_handle, cfg_file_path = self.cfg, self.logger_handle, self.cfg_file_path
        # TODO: bug occurs if use --pyt bash in slurm
        rank_id = int(os.environ['RANK'])
        # build dataset and dataloader
        dataset = BuildDataset(mode='TEST', logger_handle=logger_handle, dataset_cfg=cfg.SEGMENTOR_CFG['dataset'])
        assert dataset.num_classes == cfg.SEGMENTOR_CFG['num_classes'], 'parsed config file %s error' % cfg_file_path
        dataloader_cfg = copy.deepcopy(cfg.SEGMENTOR_CFG['dataloader'])
        dataloader_cfg['test']['batch_size'], dataloader_cfg['test']['num_workers'] = dataloader_cfg['test'].pop('batch_size_per_gpu'), dataloader_cfg['test'].pop('num_workers_per_gpu')
        dataloader = BuildDistributedDataloader(dataset=dataset, dataloader_cfg=dataloader_cfg['test'])
        # start to eval
        segmentor.eval()
        seg_results = {}
        align_corners = segmentor.module.align_corners if hasattr(segmentor, 'module') else segmentor.align_corners
        with torch.no_grad():
            dataloader.sampler.set_epoch(0)
            pbar = tqdm(enumerate(dataloader))
            for batch_idx, samples_meta in pbar:
                pbar.set_description('Processing %s/%s in rank %s' % (batch_idx+1, len(dataloader), rank_id))
                seg_logits = segmentor.module.inference(samples_meta['image']) if hasattr(segmentor, 'module') else segmentor.inference(samples_meta['image'])
                for seg_idx in range(len(seg_logits)):
                    seg_logit = F.interpolate(seg_logits[seg_idx: seg_idx+1], size=(samples_meta['height'][seg_idx], samples_meta['width'][seg_idx]), mode='bilinear', align_corners=align_corners)
                    seg_pred = (torch.argmax(seg_logit[0], dim=0)).cpu().numpy().astype(np.int32)
                    seg_gt = samples_meta['seg_target'][seg_idx].cpu().numpy().astype(np.int32)
                    seg_gt[seg_gt >= dataset.num_classes] = -1
                    seg_results[samples_meta['id'][seg_idx]] = {'seg_pred': seg_pred, 'seg_gt': seg_gt}
        dist.barrier()
        # post process
        seg_preds, seg_gts, _ = postprocesspredgtpairs(seg_results=seg_results, cfg=cfg, logger_handle=logger_handle)
        dist.barrier()
        if ismainprocess():
            result = dataset.evaluate(
                seg_preds=seg_preds, seg_gts=seg_gts, num_classes=cfg.SEGMENTOR_CFG['num_classes'], ignore_index=-1, **cfg.SEGMENTOR_CFG['inference'].get('evaluate', {}),
            )
            logger_handle.info(result, main_process_only=True)
        segmentor.train()
        dist.barrier()


'''run'''
if __name__ == '__main__':
    # parse arguments
    cmd_args = parsecmdargs()
    # instanced Trainer
    client = Trainer(cmd_args=cmd_args)
    # start
    client.start()