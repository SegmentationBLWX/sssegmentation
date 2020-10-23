'''
Function:
    load the CityScapes dataset
Author:
    Zhenchao Jin
'''
import os
import pandas as pd
from .base import *


'''CityScapes dataset'''
class CityScapesDataset(BaseDataset):
    num_classes = 19
    classnames = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', 
                  'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 
                  'truck', 'bus', 'train', 'motorcycle', 'bicycle']
    clsid2label = {
                    -1: 255, 0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
                    7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 
                    15: 255, 16: 255, 17: 5, 18: 255, 19: 6, 20: 7, 21: 8, 22: 9, 
                    23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 255, 30: 255, 
                    31: 16, 32: 17, 33: 18
                }
    assert num_classes == len(classnames)
    def __init__(self, mode, logger_handle, dataset_cfg, **kwargs):
        super(CityScapesDataset, self).__init__(mode, logger_handle, dataset_cfg, **kwargs)
        # obtain the dirs
        rootdir = dataset_cfg['rootdir']
        self.image_dir = os.path.join(rootdir, 'leftImg8bit', dataset_cfg['set'])
        self.ann_dir = os.path.join(rootdir, 'gtFine', dataset_cfg['set'])
        # obatin imageids
        df = pd.read_csv(os.path.join(rootdir, dataset_cfg['set']+'.txt'), names=['imageids'])
        self.imageids = df['imageids'].values
        self.imageids = [str(_id) for _id in self.imageids]
    '''pull item'''
    def __getitem__(self, index):
        imageid = self.imageids[index]
        imagepath = os.path.join(self.image_dir, imageid+'.png')
        annpath = os.path.join(self.ann_dir, imageid.replace('leftImg8bit', 'gtFine_labelIds')+'.png')
        sample = self.read(imagepath, annpath) if 'test' not in self.dataset_cfg['set'] else self.read(imagepath, annpath, False)
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