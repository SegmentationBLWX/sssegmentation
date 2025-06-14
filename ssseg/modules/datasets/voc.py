'''
Function:
    Implementation of VOCDataset, PascalContextDataset and PascalContext59Dataset
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
    palette = [
        (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128), (64, 0, 0),
        (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128), (0, 64, 0),
        (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)
    ]
    clsid2label = {255: -100}
    assert num_classes == len(classnames) and num_classes == len(palette)
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(VOCDataset, self).__init__(mode=mode, logger_handle=logger_handle, dataset_cfg=dataset_cfg)
        # obtain the dirs
        rootdir = dataset_cfg['rootdir']
        self.image_dir = os.path.join(rootdir, 'JPEGImages')
        self.ann_dir = os.path.join(rootdir, 'SegmentationClass')
        self.set_dir = os.path.join(rootdir, 'ImageSets', 'Segmentation')
        # obatin imageids
        df = pd.read_csv(os.path.join(self.set_dir, dataset_cfg['set']+'.txt'), names=['imageids'])
        self.imageids = df['imageids'].values
        self.imageids = [str(_id) for _id in self.imageids]


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
    palette = [
        (120, 120, 120), (180, 120, 120), (6, 230, 230), (80, 50, 50), (4, 200, 3), (120, 120, 80), (140, 140, 140), (204, 5, 255),
        (230, 230, 230), (4, 250, 7), (224, 5, 255), (235, 255, 7), (150, 5, 61), (120, 120, 70), (8, 255, 51), (255, 6, 82),
        (143, 255, 140), (204, 255, 4), (255, 51, 7), (204, 70, 3), (0, 102, 200), (61, 230, 250), (255, 6, 51), (11, 102, 255),
        (255, 7, 71), (255, 9, 224), (9, 7, 230), (220, 220, 220), (255, 9, 92), (112, 9, 255), (8, 255, 214), (7, 255, 224),
        (255, 184, 6), (10, 255, 71), (255, 41, 10), (7, 255, 255), (224, 255, 8), (102, 8, 255), (255, 61, 6), (255, 194, 7),
        (255, 122, 8), (0, 255, 20), (255, 8, 41), (255, 5, 153), (6, 51, 255), (235, 12, 255), (160, 150, 20), (0, 163, 255),
        (140, 140, 140), (250, 10, 15), (20, 255, 0), (31, 255, 0), (255, 31, 0), (255, 224, 0), (153, 255, 0), (0, 0, 255),
        (255, 71, 0), (0, 235, 255), (0, 173, 255), (31, 0, 255)
    ]
    clsid2label = {255: -100}
    assert num_classes == len(classnames) and num_classes == len(palette)
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(PascalContextDataset, self).__init__(mode=mode, logger_handle=logger_handle, dataset_cfg=dataset_cfg)
        # obtain the dirs
        rootdir = dataset_cfg['rootdir']
        self.image_dir = os.path.join(rootdir, 'JPEGImages')
        self.ann_dir = os.path.join(rootdir, 'SegmentationClassContext')
        self.set_dir = os.path.join(rootdir, 'ImageSets', 'SegmentationContext')
        # obatin imageids
        df = pd.read_csv(os.path.join(self.set_dir, dataset_cfg['set']+'.txt'), names=['imageids'])
        self.imageids = df['imageids'].values
        self.imageids = [str(_id) for _id in self.imageids]


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
    palette = [
        (180, 120, 120), (6, 230, 230), (80, 50, 50), (4, 200, 3), (120, 120, 80), (140, 140, 140), (204, 5, 255),
        (230, 230, 230), (4, 250, 7), (224, 5, 255), (235, 255, 7), (150, 5, 61), (120, 120, 70), (8, 255, 51), (255, 6, 82),
        (143, 255, 140), (204, 255, 4), (255, 51, 7), (204, 70, 3), (0, 102, 200), (61, 230, 250), (255, 6, 51), (11, 102, 255),
        (255, 7, 71), (255, 9, 224), (9, 7, 230), (220, 220, 220), (255, 9, 92), (112, 9, 255), (8, 255, 214), (7, 255, 224),
        (255, 184, 6), (10, 255, 71), (255, 41, 10), (7, 255, 255), (224, 255, 8), (102, 8, 255), (255, 61, 6), (255, 194, 7),
        (255, 122, 8), (0, 255, 20), (255, 8, 41), (255, 5, 153), (6, 51, 255), (235, 12, 255), (160, 150, 20), (0, 163, 255),
        (140, 140, 140), (250, 10, 15), (20, 255, 0), (31, 255, 0), (255, 31, 0), (255, 224, 0), (153, 255, 0), (0, 0, 255),
        (255, 71, 0), (0, 235, 255), (0, 173, 255), (31, 0, 255)
    ]
    clsid2label = {0: -100, 255: -100}
    for i in range(1, num_classes+1): clsid2label[i] = i - 1
    assert num_classes == len(classnames) and num_classes == len(palette)
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(PascalContext59Dataset, self).__init__(mode=mode, logger_handle=logger_handle, dataset_cfg=dataset_cfg)
        # obtain the dirs
        rootdir = dataset_cfg['rootdir']
        self.image_dir = os.path.join(rootdir, 'JPEGImages')
        self.ann_dir = os.path.join(rootdir, 'SegmentationClassContext')
        self.set_dir = os.path.join(rootdir, 'ImageSets', 'SegmentationContext')
        # obatin imageids
        df = pd.read_csv(os.path.join(self.set_dir, dataset_cfg['set']+'.txt'), names=['imageids'])
        self.imageids = df['imageids'].values
        self.imageids = [str(_id) for _id in self.imageids]