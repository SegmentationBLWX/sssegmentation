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
        BuildDataset, BuildSegmentor, BuildLoggerHandle, ConfigParser, touchdir, loadckpts
    )
except:
    from .modules import (
        BuildDataset, BuildSegmentor, BuildLoggerHandle, ConfigParser, touchdir, loadckpts
    )
warnings.filterwarnings('ignore')


'''parsecmdargs'''
def parsecmdargs():
    parser = argparse.ArgumentParser(description='SSSegmentation is an open source supervised semantic segmentation toolbox based on PyTorch')
    parser.add_argument('--imagedir', dest='imagedir', help='image directory, which means we let the segmentor inference on the images existed in the given image directory', type=str)
    parser.add_argument('--imagepath', dest='imagepath', help='image path, which means we let the segmentor inference on the given image', type=str)
    parser.add_argument('--outputdir', dest='outputdir', help='directory to save output image(s)', type=str, default='inference_outputs')
    parser.add_argument('--cfgfilepath', dest='cfgfilepath', help='config file path you want to use', type=str, required=True)
    parser.add_argument('--ckptspath', dest='ckptspath', help='checkpoints you want to resume from', type=str, required=True)
    parser.add_argument('--ema', dest='ema', help='please add --ema if you want to load ema weights for segmentors', default=False, action='store_true')
    args = parser.parse_args()
    return args


'''Inferencer'''
class Inferencer():
    def __init__(self):
        self.cmd_args = parsecmdargs()
        self.cfg, self.cfg_file_path = ConfigParser()(self.cmd_args.cfgfilepath)
        assert self.cmd_args.imagepath or self.cmd_args.imagedir, 'imagepath or imagedir should be specified'
        # open full fp32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    '''start'''
    def start(self):
        cmd_args, cfg, cfg_file_path = self.cmd_args, self.cfg, self.cfg_file_path
        # touch work dir and output dir
        touchdir(cfg.SEGMENTOR_CFG['work_dir'])
        touchdir(cmd_args.outputdir)
        # cuda detect
        use_cuda = torch.cuda.is_available()
        # initialize logger_handle
        logger_handle = BuildLoggerHandle(cfg.SEGMENTOR_CFG['logger_handle_cfg'])
        # build segmentor
        cfg.SEGMENTOR_CFG['backbone']['pretrained'] = False
        segmentor = BuildSegmentor(segmentor_cfg=cfg.SEGMENTOR_CFG, mode='TEST')
        if use_cuda: segmentor = segmentor.cuda()
        # build dataset
        cfg.SEGMENTOR_CFG['dataset']['evalmode'] = 'server'
        dataset = BuildDataset(mode='TEST', logger_handle=logger_handle, dataset_cfg=cfg.SEGMENTOR_CFG['dataset'])
        # build palette
        palette = dataset.palette
        # load ckpts
        cmd_args.local_rank = 0
        ckpts = loadckpts(cmd_args.ckptspath)
        try:
            segmentor.load_state_dict(ckpts['model'] if not cmd_args.ema else ckpts['model_ema'])
        except Exception as e:
            logger_handle.warning(str(e) + '\n' + 'Try to load ckpts by using strict=False')
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
            infer_tricks = inference_cfg['tricks']
            cascade_cfg = infer_tricks.get('cascade', {'key_for_pre_output': 'memory_gather_logits', 'times': 1, 'forward_default_args': None})
            sample_meta = dataset.read(imagepath)
            image = sample_meta['image']
            sample_meta = dataset.synctransforms(sample_meta)
            image_tensor = sample_meta['image'].unsqueeze(0).type(FloatTensor)
            for idx in range(cascade_cfg['times']):
                forward_args = None
                if idx > 0: 
                    output_list = [
                        F.interpolate(outputs, size=output_list[-1].shape[2:], mode='bilinear', align_corners=segmentor.align_corners) for outputs in output_list
                    ]
                    forward_args = {cascade_cfg['key_for_pre_output']: sum(output_list) / len(output_list)}
                    if cascade_cfg['forward_default_args'] is not None: 
                        forward_args.update(cascade_cfg['forward_default_args'])
                output_list = segmentor.auginference(image_tensor, forward_args)
            output_list = [
                F.interpolate(output, size=(sample_meta['height'], sample_meta['width']), mode='bilinear', align_corners=segmentor.align_corners) for output in output_list
            ]
            output = sum(output_list) / len(output_list)
            pred = (torch.argmax(output[0], dim=0)).cpu().numpy().astype(np.int32)
            mask = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
            for clsid, color in enumerate(palette):
                mask[pred == clsid, :] = np.array(color)[::-1]
            image = image * 0.5 + mask * 0.5
            image = image.astype(np.uint8)
            if cmd_args.outputdir:
                cv2.imwrite(os.path.join(cmd_args.outputdir, imagepath.split('/')[-1].split('.')[0] + '.png'), image)
            else:
                cv2.imwrite(os.path.join(cfg.SEGMENTOR_CFG['work_dir'], imagepath.split('/')[-1].split('.')[0] + '.png'), image)


'''debug'''
if __name__ == '__main__':
    with torch.no_grad():
        client = Inferencer()
        client.start()