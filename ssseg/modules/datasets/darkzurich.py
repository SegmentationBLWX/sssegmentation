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
    assert num_classes == len(classnames)
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(DarkZurichDataset, self).__init__(mode, logger_handle, dataset_cfg)
        assert dataset_cfg['set'] in ['val'], 'only support test on DarkZurichDataset'
        # obtain the dirs
        rootdir = dataset_cfg['rootdir']
        self.image_dir = os.path.join(rootdir, 'rgb_anon/val/night/GOPR0356')
        self.ann_dir = os.path.join(rootdir, 'gt/val/night/GOPR0356')
        # obatin imageids
        df = pd.read_csv(os.path.join(rootdir, 'lists_file_names/val_filenames.txt'), names=['imageids'])
        self.imageids = df['imageids'].values
        self.imageids = [str(_id) for _id in self.imageids]
    '''pull item'''
    def __getitem__(self, index):
        imageid = self.imageids[index].split('/')[-1]
        imagepath = os.path.join(self.image_dir, f'{imageid}.png')
        annpath = os.path.join(self.ann_dir, f'{imageid}.png')
        sample = self.read(imagepath, annpath, self.dataset_cfg.get('with_ann', True))
        sample.update({'id': imageid})
        if self.mode == 'TRAIN':
            sample = self.synctransform(sample, 'without_totensor_normalize_pad')
            sample['edge'] = self.generateedge(sample['segmentation'].copy())
            sample = self.synctransform(sample, 'only_totensor_normalize_pad')
        else:
            sample = self.synctransform(sample, 'all')
        return sample
    '''length'''
    def __len__(self):
        return len(self.imageids)