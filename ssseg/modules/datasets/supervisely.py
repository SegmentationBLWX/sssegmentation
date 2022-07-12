'''
Function:
    Load the supervisely dataset
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
    assert num_classes == len(classnames)
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(SuperviselyDataset, self).__init__(mode, logger_handle, dataset_cfg)
        # obtain the dirs
        rootdir = dataset_cfg['rootdir']
        self.image_dir = os.path.join(rootdir, 'Images', dataset_cfg['set'])
        self.ann_dir = os.path.join(rootdir, 'Anno-Person', dataset_cfg['set'])
        # obatin imageids
        df = pd.read_csv(os.path.join(rootdir, dataset_cfg['set']+'.txt'), names=['imageids'])
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
            sample['segmentation'][sample['segmentation'] == 255] = 1.
            sample = self.synctransform(sample, 'without_totensor_normalize_pad')
            sample['edge'] = self.generateedge(sample['segmentation'].copy())
            sample = self.synctransform(sample, 'only_totensor_normalize_pad')
        else:
            sample['groundtruth'][sample['groundtruth'] == 255] = 1.
            sample = self.synctransform(sample, 'all')
        return sample
    '''length'''
    def __len__(self):
        return len(self.imageids)