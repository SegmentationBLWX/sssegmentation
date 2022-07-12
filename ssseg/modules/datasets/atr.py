'''
Function:
    Load the ATR dataset
Author:
    Zhenchao Jin
'''
import os
import pandas as pd
from .base import BaseDataset


'''ATRDataset'''
class ATRDataset(BaseDataset):
    num_classes = 18
    classnames = [
        '__background__', 'hat', 'hair', 'sunglasses', 'coat', 'skirt', 
        'pants', 'dress', 'belt', 'leftShoe', 'rightShoe', 'face', 
        'leftLeg', 'rightLeg', 'leftHand', 'rightHand', 'bags', 'scarf'
    ]
    assert num_classes == len(classnames)
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(ATRDataset, self).__init__(mode, logger_handle, dataset_cfg)
        # obtain the dirs
        rootdir = dataset_cfg['rootdir']
        self.image_dir = os.path.join(rootdir, 'JPEGImages')
        self.ann_dir = os.path.join(rootdir, 'SegmentationClassAug')
        # obatin imageids
        df = pd.read_csv(os.path.join(rootdir, dataset_cfg['set']+'_id.txt'), names=['imageids'])
        self.imageids = df['imageids'].values
        self.imageids = [str(_id) for _id in self.imageids]
    '''pull item'''
    def __getitem__(self, index):
        imageid = self.imageids[index]
        imagepath = os.path.join(self.image_dir, imageid+'.jpg')
        annpath = os.path.join(self.ann_dir, imageid+'.png')
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