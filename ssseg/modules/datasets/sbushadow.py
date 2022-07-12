'''
Function:
    Load the sbu-shadow dataset
Author:
    Zhenchao Jin
'''
import os
from .base import BaseDataset


'''SBUShadowDataset'''
class SBUShadowDataset(BaseDataset):
    num_classes = 2
    classnames = ['__backgroud__', 'shadow']
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(SBUShadowDataset, self).__init__(mode, logger_handle, dataset_cfg)
        # obtain the dirs
        setmap_dict = {'train': 'SBUTrain4KRecoveredSmall', 'val': 'SBU-Test'}
        rootdir = dataset_cfg['rootdir']
        self.image_dir = os.path.join(rootdir, setmap_dict[dataset_cfg['set']], 'ShadowImages')
        self.ann_dir = os.path.join(rootdir, setmap_dict[dataset_cfg['set']], 'ShadowMasks')
        # obatin imageids
        self.imageids = []
        for line in open(os.path.join(rootdir, dataset_cfg['set']+'.txt'), 'r').readlines():
            if line.strip(): self.imageids.append(line.strip())
        self.imageids = [str(_id) for _id in self.imageids]
    '''pull item'''
    def __getitem__(self, index):
        imageid = self.imageids[index]
        imagepath = os.path.join(self.image_dir, imageid+'.jpg')
        annpath = os.path.join(self.ann_dir, imageid+'.png')
        sample = self.read(imagepath, annpath, self.dataset_cfg.get('with_ann', True))
        sample.update({'id': imageid})
        if self.mode == 'TRAIN':
            sample['segmentation'][sample['segmentation'] > 0] = 1.
            sample = self.synctransform(sample, 'without_totensor_normalize_pad')
            sample['edge'] = self.generateedge(sample['segmentation'].copy())
            sample = self.synctransform(sample, 'only_totensor_normalize_pad')
        else:
            sample['groundtruth'][sample['groundtruth'] > 0] = 1.
            sample = self.synctransform(sample, 'all')
        return sample
    '''length'''
    def __len__(self):
        return len(self.imageids)