'''
Function:
    demo for segmentation
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
from modules import *
from cfgs import BuildConfig
warnings.filterwarnings('ignore')


'''parse arguments in command line'''
def parseArgs():
    parser = argparse.ArgumentParser(description='sssegmentation is a general framework for our research on strongly supervised semantic segmentation')
    parser.add_argument('--imagedir', dest='imagedir', help='images dir for testing multi images', type=str)
    parser.add_argument('--imagepath', dest='imagepath', help='imagepath for testing single image', type=str)
    parser.add_argument('--outputfilename', dest='outputfilename', help='name to save output image(s)', type=str, default='')
    parser.add_argument('--cfgfilepath', dest='cfgfilepath', help='config file path you want to use', type=str, required=True)
    parser.add_argument('--checkpointspath', dest='checkpointspath', help='checkpoints you want to resume from', type=str, required=True)
    args = parser.parse_args()
    return args


'''demo for segmentation'''
class Demo():
    def __init__(self, **kwargs):
        # set attribute
        for key, value in kwargs.items(): setattr(self, key, value)
    '''start'''
    def start(self):
        # parse arguments
        cmd_args = parseArgs()
        cfg, cfg_file_path = BuildConfig(cmd_args.cfgfilepath)
        cfg.MODEL_CFG['distributed']['is_on'] = False
        cfg.MODEL_CFG['is_multi_gpus'] = False
        assert cmd_args.imagepath or cmd_args.imagedir, 'imagepath or imagedir should be specified...'
        # check backup dir
        common_cfg = cfg.COMMON_CFG['test']
        checkdir(common_cfg['backupdir'])
        # cuda detect
        use_cuda = torch.cuda.is_available()
        # initialize logger_handle
        logger_handle = Logger(common_cfg['logfilepath'])
        # instanced dataset
        dataset_cfg = cfg.DATASET_CFG['test']
        dataset = BuildDataset(mode='TEST', logger_handle=logger_handle, dataset_cfg=copy.deepcopy(cfg.DATASET_CFG), get_basedataset=True)
        palette = BuildPalette(dataset_type=dataset_cfg['type'], num_classes=dataset.num_classes, logger_handle=logger_handle)
        # instanced model
        cfg.MODEL_CFG['backbone']['pretrained'] = False
        model = BuildModel(cfg=copy.deepcopy(cfg.MODEL_CFG), mode='TEST')
        if use_cuda: model = model.cuda()
        # load checkpoints
        cmd_args.local_rank = 0
        checkpoints = loadcheckpoints(cmd_args.checkpointspath, logger_handle=logger_handle, cmd_args=cmd_args)
        try:
            model.load_state_dict(checkpoints['model'])
        except Exception as e:
            logger_handle.warning(str(e) + '\n' + 'Try to load checkpoints by using strict=False...')
            model.load_state_dict(checkpoints['model'], strict=False)
        # set eval
        model.eval()
        # start to test
        inference_cfg = copy.deepcopy(cfg.INFERENCE_CFG)
        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        if not cmd_args.imagedir:
            imagepaths = [cmd_args.imagepath]
        else:
            imagenames = os.listdir(cmd_args.imagedir)
            imagepaths = [os.path.join(cmd_args.imagedir, name) for name in imagenames]
        pbar = tqdm(range(len(imagepaths)))
        for idx in pbar:
            imagepath = imagepaths[idx]
            if imagepath.split('.')[-1] not in ['jpg', 'jpeg', 'png']: continue
            pbar.set_description('Processing %s' % imagepath)
            infer_tricks, output_list, use_probs_before_resize = inference_cfg['tricks'], [], inference_cfg['tricks']['use_probs_before_resize']
            sample = dataset.read(imagepath, 'none.png', False)
            image = sample['image']
            sample = dataset.synctransform(sample, 'all')
            image_tensor_ori = sample['image'].unsqueeze(0).type(FloatTensor)
            for scale_factor in infer_tricks['multiscale']:
                image_tensor = F.interpolate(image_tensor_ori, scale_factor=scale_factor, mode='bilinear', align_corners=model.align_corners)
                output = self.inference(model, image_tensor.type(FloatTensor), inference_cfg, cfg.MODEL_CFG['num_classes'], use_probs_before_resize)
                output_list.append(output)
                if infer_tricks['flip']:
                    image_tensor_flip = torch.from_numpy(np.flip(image_tensor.cpu().numpy(), axis=3).copy())
                    output_flip = self.inference(model, image_tensor_flip.type(FloatTensor), inference_cfg, cfg.MODEL_CFG['num_classes'], use_probs_before_resize)
                    output_flip = torch.from_numpy(np.flip(output_flip.cpu().numpy(), axis=3).copy()).type_as(output)
                    output_list.append(output_flip)
            output_list = [
                F.interpolate(output, size=(sample['height'], sample['width']), mode='bilinear', align_corners=model.align_corners) for output in output_list
            ]
            output = sum(output_list) / len(output_list)
            pred = (torch.argmax(output[0], dim=0)).cpu().numpy().astype(np.int32)
            mask = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
            for clsid, color in enumerate(palette):
                mask[pred == clsid, :] = np.array(color)[::-1]
            image = image * 0.5 + mask * 0.5
            image = image.astype(np.uint8)
            if cmd_args.outputfilename:
                cv2.imwrite(os.path.join(common_cfg['backupdir'], cmd_args.outputfilename + '_%d' % idx + '.png'), image)
            else:
                cv2.imwrite(os.path.join(common_cfg['backupdir'], imagepath.split('/')[-1].split('.')[0] + '.png'), image)
    '''inference'''
    def inference(self, model, images, inference_cfg, num_classes, use_probs_before_resize=False):
        assert inference_cfg['mode'] in ['whole', 'slide']
        if inference_cfg['mode'] == 'whole':
            if use_probs_before_resize: outputs = F.softmax(model(images), dim=1)
            else: outputs = model(images)
        else:
            assert use_probs_before_resize, 'use_probs_before_resize should be set as True when using slide mode'
            align_corners = model.align_corners if hasattr(model, 'align_corners') else model.module.align_corners
            opts = inference_cfg['opts']
            stride_h, stride_w = opts['stride']
            cropsize_h, cropsize_w = opts['cropsize']
            batch_size, _, image_h, image_w = images.size()
            num_grids_h = max(image_h - cropsize_h + stride_h - 1, 0) // stride_h + 1
            num_grids_w = max(image_w - cropsize_w + stride_w - 1, 0) // stride_w + 1
            outputs = images.new_zeros((batch_size, num_classes, image_h, image_w))
            count_mat = images.new_zeros((batch_size, 1, image_h, image_w))
            for h_idx in range(num_grids_h):
                for w_idx in range(num_grids_w):
                    x1, y1 = w_idx * stride_w, h_idx * stride_h
                    x2, y2 = min(x1 + cropsize_w, image_w), min(y1 + cropsize_h, image_h)
                    x1, y1 = max(x2 - cropsize_w, 0), max(y2 - cropsize_h, 0)
                    crop_images = images[:, :, y1:y2, x1:x2]
                    outputs_crop = F.softmax(F.interpolate(model(crop_images), size=crop_images.size()[2:], mode='bilinear', align_corners=align_corners), dim=1)
                    outputs += F.pad(outputs_crop, (int(x1), int(outputs.shape[3] - x2), int(y1), int(outputs.shape[2] - y2)))
                    count_mat[:, :, y1:y2, x1:x2] += 1
            assert (count_mat == 0).sum() == 0
            outputs = outputs / count_mat
        return outputs


'''debug'''
if __name__ == '__main__':
    with torch.no_grad():
        client = Demo()
        client.start()