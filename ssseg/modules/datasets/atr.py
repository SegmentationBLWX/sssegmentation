'''
Function:
    Implementation of ATRDataset
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
    palette = BaseDataset.randompalette(num_classes)
    assert num_classes == len(classnames) and num_classes == len(palette)
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(ATRDataset, self).__init__(mode=mode, logger_handle=logger_handle, dataset_cfg=dataset_cfg)
        # obtain the dirs
        rootdir = dataset_cfg['rootdir']
        self.image_dir = os.path.join(rootdir, 'JPEGImages')
        self.ann_dir = os.path.join(rootdir, 'SegmentationClassAug')
        # obatin imageids
        df = pd.read_csv(os.path.join(rootdir, dataset_cfg['set']+'_id.txt'), names=['imageids'])
        self.imageids = df['imageids'].values
        self.imageids = [str(_id) for _id in self.imageids]