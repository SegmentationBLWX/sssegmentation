'''
Function:
    demo for segmentation
Author:
    Zhenchao Jin
'''
import os
import cv2
import torch
import warnings
import argparse
import numpy as np
from tqdm import tqdm
from modules import *
from cfgs import BuildConfig
import torch.nn.functional as F
warnings.filterwarnings('ignore')


'''parse arguments in command line'''
def parseArgs():
    parser = argparse.ArgumentParser(description='sssegmentation is a general framework for our research on strongly supervised semantic segmentation')
    parser.add_argument('--imagedir', dest='imagedir', help='images dir for testing multi images', type=str)
    parser.add_argument('--imagepath', dest='imagepath', help='imagepath for testing single image', type=str)
    parser.add_argument('--modelname', dest='modelname', help='model you want to test', type=str, required=True)
    parser.add_argument('--backbonename', dest='backbonename', help='backbone network for testing', type=str, required=True)
    parser.add_argument('--outputfilename', dest='outputfilename', help='name to save output image(s)', type=str, default='seg_out.png')
    parser.add_argument('--checkpointspath', dest='checkpointspath', help='checkpoints you want to resume from', type=str, required=True)
    parser.add_argument('--datasetname', dest='datasetname', help='dataset you used to train, for locating the config filepath', type=str, required=True)    
    args = parser.parse_args()
    return args


'''demo for segmentation'''
def demo():
    # parse arguments
    cmd_args = parseArgs()
    cfg, cfg_file_path = BuildConfig(cmd_args.modelname, cmd_args.datasetname, cmd_args.backbonename)
    cfg.MODEL_CFG['distributed']['is_on'] = False
    cfg.MODEL_CFG['is_multi_gpus'] = False
    assert cmd_args.imagepath or cmd_args.imagedir, 'imagepath or imagedir should be specified...'
    cmd_args.outputfilename = ''.join(cmd_args.outputfilename.split('.')[:-1])
    # check backup dir
    common_cfg = cfg.COMMON_CFG['test']
    checkdir(common_cfg['backupdir'])
    # cuda detect
    use_cuda = torch.cuda.is_available()
    # initialize logger_handle
    logger_handle = Logger(common_cfg['logfilepath'])
    # instanced dataset
    dataset_cfg = cfg.DATASET_CFG['test']
    dataset = BuildDataset(mode='TEST', logger_handle=logger_handle, dataset_cfg=cfg.DATASET_CFG, get_basedataset=True)
    palette = BuildPalette(dataset_cfg['type'])
    # instanced model
    cfg.MODEL_CFG['backbone']['pretrained'] = False
    model = BuildModel(model_type=cmd_args.modelname, cfg=cfg.MODEL_CFG, mode='TEST')
    if use_cuda: model = model.cuda()
    # load checkpoints
    cmd_args.local_rank = 0
    checkpoints = loadcheckpoints(cmd_args.checkpointspath, logger_handle=logger_handle, cmd_args=cmd_args)
    model.load_state_dict(checkpoints['model'])    
    # set eval
    model.eval()
    # start to test
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
        sample = dataset.read(imagepath, '', False)
        image = sample['image']
        sample = dataset.synctransform(sample, 'all')
        output = model(sample['image'].unsqueeze(0).type(FloatTensor))
        pred = F.interpolate(output, size=(sample['height'], sample['width']), mode='bilinear', align_corners=model.align_corners)[0]
        pred = (torch.argmax(pred, dim=0)).cpu().numpy().astype(np.int32)
        mask = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for clsid, color in enumerate(palette):
            mask[pred == clsid, :] = np.array(color)
        image = image * 0.5 + mask * 0.5
        image = image.astype(np.uint8)    
        cv2.imwrite(os.path.join(common_cfg['backupdir'], cmd_args.outputfilename + '_%d' % idx + '.png'), image)


'''debug'''
if __name__ == '__main__':
    demo()