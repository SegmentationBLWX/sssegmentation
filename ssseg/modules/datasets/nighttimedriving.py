'''
Function:
    Implementation of NighttimeDrivingDataset
Author:
    Zhenchao Jin
'''
import os
from .base import BaseDataset


'''NighttimeDrivingDataset'''
class NighttimeDrivingDataset(BaseDataset):
    num_classes = 19
    classnames = [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', 
        'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 
        'truck', 'bus', 'train', 'motorcycle', 'bicycle'
    ]
    palette = [
        (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
        (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100),
        (0, 80, 100), (0, 0, 230), (119, 11, 32)
    ]
    assert num_classes == len(classnames) and num_classes == len(palette)
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(NighttimeDrivingDataset, self).__init__(mode=mode, logger_handle=logger_handle, dataset_cfg=dataset_cfg)
        assert dataset_cfg['set'] in ['val'], 'only support testing on NighttimeDrivingDataset'
        # obtain the dirs
        rootdir = dataset_cfg['rootdir']
        self.image_dir = os.path.join(rootdir, 'leftImg8bit/test/night')
        self.ann_dir = os.path.join(rootdir, 'gtCoarse_daytime_trainvaltest/test/night')
        # obatin imageids
        self.imageids = os.listdir(self.image_dir)
        self.imageids = [_id.replace('_leftImg8bit.png', '') for _id in self.imageids]
        self.ann_ext = '_gtCoarse_labelTrainIds.png'
        self.image_ext = '_leftImg8bit.png'