'''
Function:
    Implementation of STAREDataset
Author:
    Zhenchao Jin
'''
import os
import pandas as pd
from .base import BaseDataset


'''STAREDataset'''
class STAREDataset(BaseDataset):
    num_classes = 2
    classnames = ['__background__', 'vessel']
    palette = [(0, 0, 0), (255, 0, 0)]
    assert num_classes == len(classnames) and num_classes == len(palette)
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(STAREDataset, self).__init__(mode=mode, logger_handle=logger_handle, dataset_cfg=dataset_cfg)
        # obtain the dirs
        setmap_dict = {'train': 'training', 'val': 'validation'}
        rootdir = dataset_cfg['rootdir']
        self.image_dir = os.path.join(rootdir, 'images', setmap_dict[dataset_cfg['set']])
        self.ann_dir = os.path.join(rootdir, 'annotations', setmap_dict[dataset_cfg['set']])
        # obatin imageids
        df = pd.read_csv(os.path.join(rootdir, dataset_cfg['set']+'.txt'), names=['imageids'])
        self.imageids = df['imageids'].values
        self.imageids = [str(_id) for _id in self.imageids]
        self.ann_ext = '.ah.png'
        self.image_ext = self.imageids[0].split('.')[-1]
        self.imageids = [_id.split('.')[0] for _id in self.imageids]