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
import pickle
import warnings
import argparse
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
from configs import BuildConfig
from modules import (
    BuildDataset, BuildDistributedDataloader, BuildDistributedModel, BuildSegmentor, Logger, 
    initslurm, touchdir, loadckpts, saveckpts, BuildOptimizer, BuildScheduler, judgefileexist
)
warnings.filterwarnings('ignore')


'''parse arguments in command line'''
def parsecmdargs():
    parser = argparse.ArgumentParser(description='SSSegmentation is an open source supervised semantic segmentation toolbox based on PyTorch')
    parser.add_argument('--local_rank', '--local-rank', dest='local_rank', help='node rank for distributed training', default=0, type=int)
    parser.add_argument('--nproc_per_node', dest='nproc_per_node', help='number of process per node', default=8, type=int)
    parser.add_argument('--cfgfilepath', dest='cfgfilepath', help='config file path you want to use', type=str, required=True)
    parser.add_argument('--ckptspath', dest='ckptspath', help='checkpoints you want to resume from', default='', type=str)
    parser.add_argument('--slurm', dest='slurm', help='please add --slurm if you are using slurm', default=False, action='store_true')
    args = parser.parse_args()
    if torch.__version__.startswith('2.'):
        args.local_rank = int(os.environ['LOCAL_RANK'])
    if args.slurm: initslurm(args, str(8888 + random.randint(0, 1000)))
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
        # open full fp32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    '''start trainer'''
    def start(self):
        cfg, ngpus_per_node, logger_handle, cmd_args, cfg_file_path = self.cfg, self.ngpus_per_node, self.logger_handle, self.cmd_args, self.cfg_file_path
        # build dataset and dataloader
        dataset = BuildDataset(mode='TRAIN', logger_handle=logger_handle, dataset_cfg=cfg.SEGMENTOR_CFG['dataset'])
        assert dataset.num_classes == cfg.SEGMENTOR_CFG['num_classes'], 'parsed config file %s error' % cfg_file_path
        dataloader_cfg = copy.deepcopy(cfg.SEGMENTOR_CFG['dataloader'])
        auto_adapt_to_expected_train_bs = dataloader_cfg.pop('auto_adapt_to_expected_train_bs')
        expected_total_train_bs_for_assert = dataloader_cfg.pop('expected_total_train_bs_for_assert')
        if auto_adapt_to_expected_train_bs:
            dataloader_cfg['train']['batch_size_per_gpu'] = expected_total_train_bs_for_assert // ngpus_per_node
        dataloader_cfg['train']['batch_size'], dataloader_cfg['train']['num_workers'] = dataloader_cfg['train'].pop('batch_size_per_gpu'), dataloader_cfg['train'].pop('num_workers_per_gpu')
        dataloader_cfg['test']['batch_size'], dataloader_cfg['test']['num_workers'] = dataloader_cfg['test'].pop('batch_size_per_gpu'), dataloader_cfg['test'].pop('num_workers_per_gpu')
        assert expected_total_train_bs_for_assert == dataloader_cfg['train']['batch_size'] * ngpus_per_node
        dataloader = BuildDistributedDataloader(dataset=dataset, dataloader_cfg=dataloader_cfg['train'])
        # build segmentor
        segmentor = BuildSegmentor(segmentor_cfg=copy.deepcopy(cfg.SEGMENTOR_CFG), mode='TRAIN')
        torch.cuda.set_device(cmd_args.local_rank)
        segmentor.cuda(cmd_args.local_rank)
        torch.backends.cudnn.benchmark = cfg.SEGMENTOR_CFG['benchmark']
        # build optimizer
        optimizer = BuildOptimizer(segmentor, cfg.SEGMENTOR_CFG['scheduler']['optimizer'])
        # build fp16
        fp16_cfg = self.cfg.SEGMENTOR_CFG.get('fp16_cfg', {'type': None})
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
                logger_handle.warning(str(e) + '\n' + 'Try to load ckpts by using strict=False')
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
        else:
            cmd_args.ckptspath = ''
        # parallel segmentor
        build_dist_model_cfg = self.cfg.SEGMENTOR_CFG.get('build_dist_model_cfg', {})
        build_dist_model_cfg.update({'device_ids': [cmd_args.local_rank]})
        segmentor = BuildDistributedModel(segmentor, build_dist_model_cfg)
        if fp16_type in ['pytorch']:
            segmentor.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
        # print config
        if (cmd_args.local_rank == 0) and (int(os.environ.get('SLURM_PROCID', 0)) == 0):
            logger_handle.info(f'Config file path: {cfg_file_path}')
            logger_handle.info(f'Config details: \n{cfg.SEGMENTOR_CFG}')
            logger_handle.info(f'Resume from: {cmd_args.ckptspath}')
        # start to train the segmentor
        FloatTensor, losses_log_dict_memory = torch.cuda.FloatTensor, {}
        for epoch in range(start_epoch, end_epoch+1):
            # --set train
            segmentor.train()
            dataloader.sampler.set_epoch(epoch)
            # --train epoch
            for batch_idx, samples_meta in enumerate(dataloader):
                learning_rate = scheduler.updatelr()
                images = samples_meta['image'].type(FloatTensor)
                targets = {'seg_target': samples_meta['seg_target'].type(FloatTensor)}
                if 'edge_target' in samples_meta: targets['edge_target'] = samples_meta['edge_target'].type(FloatTensor)
                optimizer.zero_grad()
                forward_kwargs = {'learning_rate': learning_rate, 'epoch': epoch} if cfg.SEGMENTOR_CFG['type'] in ['MCIBI', 'MCIBIPlusPlus'] else {}
                if fp16_type in ['pytorch']:
                    with autocast(**fp16_cfg['autocast']):
                        loss, losses_log_dict = segmentor(images, targets, **forward_kwargs)
                else:
                    loss, losses_log_dict = segmentor(images, targets, **forward_kwargs)
                for key, value in losses_log_dict.items():
                    if key in losses_log_dict_memory: 
                        losses_log_dict_memory[key].append(value)
                    else: 
                        losses_log_dict_memory[key] = [value]
                if fp16_type in ['apex']:
                    with apex.amp.scale_loss(loss, optimizer, **fp16_cfg['scale_loss']) as scaled_loss:
                        scaled_loss.backward()
                elif fp16_type in ['pytorch']:
                    grad_scaler.scale(loss).backward()
                else:
                    loss.backward()
                scheduler.step(grad_scaler=grad_scaler)
                if (cmd_args.local_rank == 0) and (scheduler.cur_iter % cfg.SEGMENTOR_CFG['log_interval_iterations'] == 0) and (int(os.environ.get('SLURM_PROCID', 0)) == 0):
                    print_log = {
                        'cur_epoch': epoch, 'max_epochs': end_epoch, 'cur_iter': scheduler.cur_iter, 'max_iters': scheduler.max_iters,
                        'cur_iter_in_cur_epoch': batch_idx+1, 'max_iters_in_cur_epoch': len(dataloader), 'segmentor': cfg.SEGMENTOR_CFG['type'], 
                        'backbone': cfg.SEGMENTOR_CFG['backbone']['structure_type'], 'dataset': cfg.SEGMENTOR_CFG['dataset']['type'], 
                        'learning_rate': learning_rate,
                    }
                    for key in list(losses_log_dict_memory.keys()):
                        print_log[key] = sum(losses_log_dict_memory[key]) / len(losses_log_dict_memory[key])
                    logger_handle.info(print_log)
                    losses_log_dict_memory = dict()
            scheduler.cur_epoch = epoch
            # --save ckpts
            if (epoch % cfg.SEGMENTOR_CFG['save_interval_epochs'] == 0) or (epoch == end_epoch):
                state_dict = scheduler.state()
                state_dict['model'] = segmentor.module.state_dict()
                if fp16_type in ['apex']:
                    state_dict['amp'] = apex.amp.state_dict()
                elif fp16_type in ['pytorch']:
                    state_dict['grad_scaler'] = grad_scaler.state_dict()
                savepath = os.path.join(cfg.SEGMENTOR_CFG['work_dir'], 'epoch_%s.pth' % epoch)
                if (cmd_args.local_rank == 0) and (int(os.environ.get('SLURM_PROCID', 0)) == 0):
                    saveckpts(state_dict, savepath)
            # --eval ckpts
            if (epoch % cfg.SEGMENTOR_CFG['eval_interval_epochs'] == 0) or (epoch == end_epoch):
                if (cmd_args.local_rank == 0) and (int(os.environ.get('SLURM_PROCID', 0)) == 0): 
                    self.logger_handle.info(f'Evaluate {cfg.SEGMENTOR_CFG["type"]} at epoch {epoch}')
                self.evaluate(segmentor)
    '''evaluate'''
    def evaluate(self, segmentor):
        cfg, ngpus_per_node, logger_handle, cmd_args, cfg_file_path = self.cfg, self.ngpus_per_node, self.logger_handle, self.cmd_args, self.cfg_file_path
        # TODO: bug occurs if use --pyt bash
        rank_id = int(os.environ['SLURM_PROCID']) if 'SLURM_PROCID' in os.environ else cmd_args.local_rank
        # build dataset and dataloader
        dataset = BuildDataset(mode='TEST', logger_handle=logger_handle, dataset_cfg=cfg.SEGMENTOR_CFG['dataset'])
        assert dataset.num_classes == cfg.SEGMENTOR_CFG['num_classes'], 'parsed config file %s error' % cfg_file_path
        dataloader_cfg = copy.deepcopy(cfg.SEGMENTOR_CFG['dataloader'])
        dataloader_cfg['test']['batch_size'], dataloader_cfg['test']['num_workers'] = dataloader_cfg['test'].pop('batch_size_per_gpu'), dataloader_cfg['test'].pop('num_workers_per_gpu')
        dataloader = BuildDistributedDataloader(dataset=dataset, dataloader_cfg=dataloader_cfg['test'])
        # start to eval
        segmentor.eval()
        segmentor.module.mode = 'TEST'
        inference_cfg, all_preds, all_gts = cfg.SEGMENTOR_CFG['inference'], [], []
        align_corners = segmentor.module.align_corners
        FloatTensor = torch.cuda.FloatTensor
        use_probs_before_resize = inference_cfg['tricks']['use_probs_before_resize']
        assert inference_cfg['mode'] in ['whole', 'slide']
        with torch.no_grad():
            dataloader.sampler.set_epoch(0)
            pbar = tqdm(enumerate(dataloader))
            for batch_idx, samples_meta in pbar:
                pbar.set_description('Processing %s/%s in rank %s' % (batch_idx+1, len(dataloader), rank_id))
                imageids, images, widths, heights, gts = samples_meta['id'], samples_meta['image'].type(FloatTensor), samples_meta['width'], samples_meta['height'], samples_meta['seg_target']
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
        filename = cfg.SEGMENTOR_CFG['resultsavepath'].split('/')[-1].split('.')[0] + f'_{rank_id}.' + cfg.SEGMENTOR_CFG['resultsavepath'].split('.')[-1]
        with open(os.path.join(cfg.SEGMENTOR_CFG['work_dir'], filename), 'wb') as fp:
            pickle.dump([all_preds, all_gts], fp)
        rank = torch.tensor([rank_id], device='cuda')
        rank_list = [rank.clone() for _ in range(ngpus_per_node)]
        dist.all_gather(rank_list, rank)
        logger_handle.info('Rank %s finished' % int(rank.item()))
        if rank_id == 0:
            all_preds_gather, all_gts_gather = [], []
            for rank in rank_list:
                rank = str(int(rank.item()))
                filename = cfg.SEGMENTOR_CFG['resultsavepath'].split('/')[-1].split('.')[0] + f'_{rank}.' + cfg.SEGMENTOR_CFG['resultsavepath'].split('.')[-1]
                fp = open(os.path.join(cfg.SEGMENTOR_CFG['work_dir'], filename), 'rb')
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
                seg_preds=all_preds, 
                seg_targets=all_gts, 
                metric_list=inference_cfg.get('metric_list', ['iou', 'miou']),
                num_classes=cfg.SEGMENTOR_CFG['num_classes'],
                ignore_index=-1,
            )
            logger_handle.info(result)
        segmentor.train()
        segmentor.module.mode = 'TRAIN'


'''main'''
def main():
    # parse arguments
    args = parsecmdargs()
    cfg, cfg_file_path = BuildConfig(args.cfgfilepath)
    # touch work dir
    touchdir(cfg.SEGMENTOR_CFG['work_dir'])
    # initialize logger_handle
    logger_handle = Logger(cfg.SEGMENTOR_CFG['logfilepath'])
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