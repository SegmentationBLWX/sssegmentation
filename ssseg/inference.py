'''
Function:
    Implementation of Inferencer
Author:
    Zhenchao Jin
'''
import os
import cv2
import copy
import torch
import warnings
import argparse
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
try:
    from modules import (
        BuildDataset, BuildSegmentor, BuildLoggerHandle, ConfigParser, touchdirs, loadckpts
    )
except:
    from .modules import (
        BuildDataset, BuildSegmentor, BuildLoggerHandle, ConfigParser, touchdirs, loadckpts
    )
warnings.filterwarnings('ignore')


'''parsecmdargs'''
def parsecmdargs():
    parser = argparse.ArgumentParser(description='SSSegmentation is an open source supervised semantic segmentation toolbox based on PyTorch.')
    parser.add_argument('--imagedir', dest='imagedir', help='Directory containing images for inference by the segmentor.', type=str)
    parser.add_argument('--imagepath', dest='imagepath', help='Path to the image for inference by the segmentor.', type=str)
    parser.add_argument('--outputdir', dest='outputdir', help='Destination directory for saving the output image(s).', type=str, default='inference_outputs')
    parser.add_argument('--cfgfilepath', dest='cfgfilepath', help='The config file path which is used to customize segmentors.', type=str, required=True)
    parser.add_argument('--ckptspath', dest='ckptspath', help='Specify the checkpoint to use for inference.', type=str, required=True)
    parser.add_argument('--ema', dest='ema', help='Please add --ema if you want to load ema weights of segmentors for inference.', default=False, action='store_true')
    cmd_args = parser.parse_args()
    return cmd_args


'''Inferencer'''
class Inferencer():
    def __init__(self, cmd_args):
        self.cmd_args = cmd_args
        self.cfg, self.cfg_file_path = ConfigParser()(self.cmd_args.cfgfilepath)
        assert self.cmd_args.imagepath or self.cmd_args.imagedir, 'imagepath or imagedir should be specified'
        # open full fp32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    '''start'''
    def start(self):
        # initialize necessary variables
        cmd_args, cfg = self.cmd_args, self.cfg
        # touch work dir and output dir
        touchdirs(cfg.SEGMENTOR_CFG['work_dir'])
        touchdirs(cmd_args.outputdir)
        # cuda detect
        use_cuda = torch.cuda.is_available()
        # initialize logger_handle
        logger_handle = BuildLoggerHandle(cfg.SEGMENTOR_CFG['logger_handle_cfg'])
        # build segmentor
        cfg.SEGMENTOR_CFG['backbone']['pretrained'] = False
        segmentor = BuildSegmentor(segmentor_cfg=cfg.SEGMENTOR_CFG, mode='TEST')
        if use_cuda: segmentor = segmentor.cuda()
        # build dataset
        cfg.SEGMENTOR_CFG['dataset']['test']['eval_env'] = 'server'
        dataset = BuildDataset(mode='TEST', logger_handle=logger_handle, dataset_cfg=cfg.SEGMENTOR_CFG['dataset'])
        # build palette
        palette = dataset.palette
        # load ckpts
        ckpts = loadckpts(cmd_args.ckptspath)
        try:
            segmentor.load_state_dict(ckpts['model'] if not cmd_args.ema else ckpts['model_ema'])
        except Exception as e:
            logger_handle.warning(str(e) + '\n' + 'Try to load ckpts by using strict=False', main_process_only=True)
            segmentor.load_state_dict(ckpts['model'] if not cmd_args.ema else ckpts['model_ema'], strict=False)
        # set eval
        segmentor.eval()
        # start to test
        inference_cfg = copy.deepcopy(cfg.SEGMENTOR_CFG['inference'])
        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        if not cmd_args.imagedir:
            imagepaths = [cmd_args.imagepath]
        else:
            imagenames = os.listdir(cmd_args.imagedir)
            imagepaths = [os.path.join(cmd_args.imagedir, name) for name in imagenames]
        pbar = tqdm(range(len(imagepaths)))
        for idx in pbar:
            imagepath = imagepaths[idx]
            if imagepath.split('.')[-1] not in ['jpg', 'jpeg', 'png']: 
                continue
            pbar.set_description('Processing %s' % imagepath)
            infer_tta_cfg = inference_cfg['tta']
            cascade_cfg = infer_tta_cfg.get('cascade', {'key_for_pre_output': 'memory_gather_logits', 'times': 1, 'forward_default_args': None})
            sample_meta = dataset.read(imagepath)
            image = sample_meta['image']
            sample_meta = dataset.synctransforms(sample_meta)
            image_tensor = sample_meta['image'].unsqueeze(0).type(FloatTensor)
            for idx in range(cascade_cfg['times']):
                forward_args = None
                if idx > 0: 
                    seg_logits_list = [
                        F.interpolate(seg_logits, size=seg_logits_list[-1].shape[2:], mode='bilinear', align_corners=segmentor.align_corners) for seg_logits in seg_logits_list
                    ]
                    forward_args = {cascade_cfg['key_for_pre_output']: sum(seg_logits_list) / len(seg_logits_list)}
                    if cascade_cfg['forward_default_args'] is not None: 
                        forward_args.update(cascade_cfg['forward_default_args'])
                seg_logits_list = segmentor.auginference(image_tensor, forward_args)
            seg_logits_list = [
                F.interpolate(seg_logits, size=(sample_meta['height'], sample_meta['width']), mode='bilinear', align_corners=segmentor.align_corners) for seg_logits in seg_logits_list
            ]
            seg_logits = sum(seg_logits_list) / len(seg_logits_list)
            seg_pred = (torch.argmax(seg_logits[0], dim=0)).cpu().numpy().astype(np.int32)
            seg_mask = np.zeros((seg_pred.shape[0], seg_pred.shape[1], 3), dtype=np.uint8)
            for clsid, color in enumerate(palette):
                seg_mask[seg_pred == clsid, :] = np.array(color)[::-1]
            image = image * 0.5 + seg_mask * 0.5
            image = image.astype(np.uint8)
            if cmd_args.outputdir:
                cv2.imwrite(os.path.join(cmd_args.outputdir, imagepath.split('/')[-1].split('.')[0] + '.png'), image)
            else:
                cv2.imwrite(os.path.join(cfg.SEGMENTOR_CFG['work_dir'], imagepath.split('/')[-1].split('.')[0] + '.png'), image)


'''run'''
if __name__ == '__main__':
    with torch.no_grad():
        # parse arguments
        cmd_args = parsecmdargs()
        # instanced Inferencer
        client = Inferencer(cmd_args=cmd_args)
        # start
        client.start()