'''
Function:
    Implementation of HRFDataset
Author:
    Zhenchao Jin
'''
import os
import pandas as pd
from .base import BaseDataset


'''HRFDataset'''
class HRFDataset(BaseDataset):
    num_classes = 2
    classnames = ['__background__', 'vessel']
    palette = [(0, 0, 0), (255, 0, 0)]
    assert num_classes == len(classnames) and num_classes == len(palette)
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(HRFDataset, self).__init__(mode=mode, logger_handle=logger_handle, dataset_cfg=dataset_cfg)
        # obtain the dirs
        setmap_dict = {'train': 'training', 'val': 'validation'}
        rootdir = dataset_cfg['rootdir']
        self.image_dir = os.path.join(rootdir, 'images', setmap_dict[dataset_cfg['set']])
        self.ann_dir = os.path.join(rootdir, 'annotations', setmap_dict[dataset_cfg['set']])
        # obatin imageids
        df = pd.read_csv(os.path.join(rootdir, dataset_cfg['set']+'.txt'), names=['imageids'])
        self.imageids = df['imageids'].values
        self.imageids = [str(_id) for _id in self.imageids]
        self.ann_ext = ''
        self.image_ext = ''