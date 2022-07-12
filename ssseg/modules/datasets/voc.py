'''
Function:
    Load the voc dataset
Author:
    Zhenchao Jin
'''
import os
import pandas as pd
from .base import BaseDataset


'''VOCDataset'''
class VOCDataset(BaseDataset):
    num_classes = 21
    classnames = [
        '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    assert num_classes == len(classnames)
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(VOCDataset, self).__init__(mode, logger_handle, dataset_cfg)
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

 
'''PascalContextDataset'''
class PascalContextDataset(BaseDataset):
    num_classes = 60
    classnames = [
        '__background__', 'aeroplane', 'bag', 'bed', 'bedclothes', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle', 'building', 'bus',
        'cabinet', 'car', 'cat', 'ceiling', 'chair', 'cloth', 'computer', 'cow', 'cup', 'curtain', 'dog', 'door', 'fence',
        'floor', 'flower', 'food', 'grass', 'ground', 'horse', 'keyboard', 'light', 'motorbike', 'mountain', 'mouse', 'person',
        'plate', 'platform', 'pottedplant', 'road', 'rock', 'sheep', 'shelves', 'sidewalk', 'sign', 'sky', 'snow', 'sofa', 'table',
        'track', 'train', 'tree', 'truck', 'tvmonitor', 'wall', 'water', 'window', 'wood'
    ]
    assert num_classes == len(classnames)
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(PascalContextDataset, self).__init__(mode, logger_handle, dataset_cfg)
        # obtain the dirs
        rootdir = dataset_cfg['rootdir']
        self.image_dir = os.path.join(rootdir, 'JPEGImages')
        self.ann_dir = os.path.join(rootdir, 'SegmentationClassContext')
        self.set_dir = os.path.join(rootdir, 'ImageSets', 'SegmentationContext')
        # obatin imageids
        df = pd.read_csv(os.path.join(self.set_dir, dataset_cfg['set']+'.txt'), names=['imageids'])
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


'''PascalContext59Dataset'''
class PascalContext59Dataset(BaseDataset):
    num_classes = 59
    classnames = [
        'aeroplane', 'bag', 'bed', 'bedclothes', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle', 'building', 'bus',
        'cabinet', 'car', 'cat', 'ceiling', 'chair', 'cloth', 'computer', 'cow', 'cup', 'curtain', 'dog', 'door', 'fence',
        'floor', 'flower', 'food', 'grass', 'ground', 'horse', 'keyboard', 'light', 'motorbike', 'mountain', 'mouse', 'person',
        'plate', 'platform', 'pottedplant', 'road', 'rock', 'sheep', 'shelves', 'sidewalk', 'sign', 'sky', 'snow', 'sofa', 'table',
        'track', 'train', 'tree', 'truck', 'tvmonitor', 'wall', 'water', 'window', 'wood'
    ]
    clsid2label = {0: 255}
    for i in range(1, num_classes+1): clsid2label[i] = i - 1
    assert num_classes == len(classnames)
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(PascalContext59Dataset, self).__init__(mode, logger_handle, dataset_cfg)
        # obtain the dirs
        rootdir = dataset_cfg['rootdir']
        self.image_dir = os.path.join(rootdir, 'JPEGImages')
        self.ann_dir = os.path.join(rootdir, 'SegmentationClassContext')
        self.set_dir = os.path.join(rootdir, 'ImageSets', 'SegmentationContext')
        # obatin imageids
        df = pd.read_csv(os.path.join(self.set_dir, dataset_cfg['set']+'.txt'), names=['imageids'])
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