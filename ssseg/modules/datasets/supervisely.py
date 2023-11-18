'''
Function:
    Implementation of SuperviselyDataset
Author:
    Zhenchao Jin
'''
import os
import pandas as pd
from .base import BaseDataset


'''SuperviselyDataset'''
class SuperviselyDataset(BaseDataset):
    num_classes = 2
    classnames = ['__background__', 'person']
    palette = [(0, 0, 0), (255, 0, 0)]
    clsid2label = {255: 1}
    assert num_classes == len(classnames) and num_classes == len(palette)
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(SuperviselyDataset, self).__init__(mode=mode, logger_handle=logger_handle, dataset_cfg=dataset_cfg)
        # obtain the dirs
        rootdir = dataset_cfg['rootdir']
        self.image_dir = os.path.join(rootdir, 'Images', dataset_cfg['set'])
        self.ann_dir = os.path.join(rootdir, 'Anno-Person', dataset_cfg['set'])
        # obatin imageids
        df = pd.read_csv(os.path.join(rootdir, dataset_cfg['set']+'.txt'), names=['imageids'])
        self.imageids = df['imageids'].values
        self.imageids = [str(_id) for _id in self.imageids]