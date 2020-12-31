'''
Function:
    load the voc dataset
Author:
    Zhenchao Jin
'''
import os
import pandas as pd
from .base import *


'''voc dataset'''
class VOCDataset(BaseDataset):
    num_classes = 21
    classnames = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                  'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                  'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    assert num_classes == len(classnames)
    def __init__(self, mode, logger_handle, dataset_cfg, **kwargs):
        super(VOCDataset, self).__init__(mode, logger_handle, dataset_cfg, **kwargs)
        # obtain the dirs
        rootdir = dataset_cfg['rootdir']
        self.image_dir = os.path.join(rootdir, 'JPEGImages')
        self.ann_dir = os.path.join(rootdir, 'Annotations')
        self.segclass_dir = os.path.join(rootdir, 'SegmentationClass')
        self.set_dir = os.path.join(rootdir, 'ImageSets', 'Segmentation')
        # obatin imageids
        df = pd.read_csv(os.path.join(self.set_dir, dataset_cfg['set']+'.txt'), names=['imageids'])
        self.imageids = df['imageids'].values
        self.imageids = [str(_id) for _id in self.imageids]
    '''pull item'''
    def __getitem__(self, index):
        imageid = self.imageids[index]
        imagepath = os.path.join(self.image_dir, imageid+'.jpg')
        annpath = os.path.join(self.segclass_dir, imageid+'.png')
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

 
'''voc context dataset'''
class VOCContextDataset(BaseDataset):
    num_classes = 60
    classnames = ['__backgroud__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'table',
                  'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'bag', 'bed',
                  'bench', 'book', 'building', 'cabinet', 'ceiling', 'cloth', 'computer', 'cup', 'door', 'fence', 'floor', 'flower',
                  'food', 'grass', 'ground', 'keyboard', 'light', 'mountain', 'mouse', 'curtain', 'platform', 'sign', 'plate', 'road',
                  'rock', 'shelves', 'sidewalk', 'sky', 'snow', 'bedclothes', 'track', 'tree', 'truck', 'wall', 'water', 'window', 'wood']
    assert num_classes == len(classnames)
    def __init__(self, mode, logger_handle, dataset_cfg, **kwargs):
        super(VOCContextDataset, self).__init__(mode, logger_handle, dataset_cfg, **kwargs)
    '''pull item'''
    def __getitem__(self, index):
        raise NotImplementedError('not be implemented')
    '''length'''
    def __len__(self):
        return len(self.imageids)