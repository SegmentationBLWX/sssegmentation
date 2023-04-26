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
    assert num_classes == len(classnames)
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(NighttimeDrivingDataset, self).__init__(mode, logger_handle, dataset_cfg)
        assert dataset_cfg['set'] in ['val'], 'only support testing on NighttimeDrivingDataset'
        # obtain the dirs
        rootdir = dataset_cfg['rootdir']
        self.image_dir = os.path.join(rootdir, 'leftImg8bit/test/night')
        self.ann_dir = os.path.join(rootdir, 'gtCoarse_daytime_trainvaltest/test/night')
        # obatin imageids
        self.imageids = os.listdir(self.ann_dir)
        self.imageids = [_id.replace('_leftImg8bit.png', '') for _id in self.imageids]
    '''pull item'''
    def __getitem__(self, index):
        imageid = self.imageids[index].split('/')[-1]
        imagepath = os.path.join(self.image_dir, f'{imageid}_leftImg8bit.png')
        annpath = os.path.join(self.ann_dir, f'{imageid}_gtCoarse_labelTrainIds.png')
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