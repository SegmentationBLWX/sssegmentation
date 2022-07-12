'''
Function:
    Load the drive dataset
Author:
    Zhenchao Jin
'''
import os
import pandas as pd
from .base import BaseDataset


'''DRIVEDataset'''
class DRIVEDataset(BaseDataset):
    num_classes = 2
    classnames = ['__background__', 'vessel']
    assert num_classes == len(classnames)
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(DRIVEDataset, self).__init__(mode, logger_handle, dataset_cfg)
        self.repeat_times = dataset_cfg.get('repeat_times', 1)
        # obtain the dirs
        setmap_dict = {'train': 'training', 'val': 'validation'}
        rootdir = dataset_cfg['rootdir']
        self.image_dir = os.path.join(rootdir, 'images', setmap_dict[dataset_cfg['set']])
        self.ann_dir = os.path.join(rootdir, 'annotations', setmap_dict[dataset_cfg['set']])
        # obatin imageids
        df = pd.read_csv(os.path.join(rootdir, dataset_cfg['set']+'.txt'), names=['imageids'])
        self.imageids = df['imageids'].values
        self.imageids = [str(_id) for _id in self.imageids]
    '''pull item'''
    def __getitem__(self, index):
        imageid = self.imageids[index % len(self.imageids)]
        imagepath = os.path.join(self.image_dir, imageid)
        annpath = os.path.join(self.ann_dir, imageid.split('.')[0] + '_manual1.png')
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
        return len(self.imageids) * self.repeat_times