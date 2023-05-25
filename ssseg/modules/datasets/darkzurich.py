'''
Function:
    Implementation of DarkZurichDataset
Author:
    Zhenchao Jin
'''
import os
import pandas as pd
from .base import BaseDataset


'''DarkZurichDataset'''
class DarkZurichDataset(BaseDataset):
    num_classes = 19
    classnames = [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', 
        'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 
        'truck', 'bus', 'train', 'motorcycle', 'bicycle'
    ]
    palette = [
        (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
        (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100),
        (0, 80, 100), (0, 0, 230), (119, 11, 32)
    ]
    assert num_classes == len(classnames) and num_classes == len(palette)
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(DarkZurichDataset, self).__init__(mode=mode, logger_handle=logger_handle, dataset_cfg=dataset_cfg)
        assert dataset_cfg['set'] in ['val'], 'only support testing on DarkZurichDataset'
        # obtain the dirs
        rootdir = dataset_cfg['rootdir']
        self.image_dir = os.path.join(rootdir, 'rgb_anon/val/night/GOPR0356')
        self.ann_dir = os.path.join(rootdir, 'gt/val/night/GOPR0356')
        # obatin imageids
        df = pd.read_csv(os.path.join(rootdir, 'lists_file_names/val_filenames.txt'), names=['imageids'])
        self.imageids = df['imageids'].values
        self.imageids = [str(_id).split('/')[-1] for _id in self.imageids]
        self.ann_ext = '_gt_labelTrainIds.png'
        self.image_ext = '_rgb_anon.png'