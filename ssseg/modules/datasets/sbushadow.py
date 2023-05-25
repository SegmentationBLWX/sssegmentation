'''
Function:
    Implementation of SBUShadowDataset
Author:
    Zhenchao Jin
'''
import os
from .base import BaseDataset


'''SBUShadowDataset'''
class SBUShadowDataset(BaseDataset):
    num_classes = 2
    classnames = ['__backgroud__', 'shadow']
    palette = [(0, 0, 0), (255, 0, 0)]
    assert num_classes == len(classnames) and num_classes == len(palette)
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(SBUShadowDataset, self).__init__(mode=mode, logger_handle=logger_handle, dataset_cfg=dataset_cfg)
        # obtain the dirs
        rootdir = dataset_cfg['rootdir']
        setmap_dict = {'train': 'SBUTrain4KRecoveredSmall', 'val': 'SBU-Test'}
        self.image_dir = os.path.join(rootdir, setmap_dict[dataset_cfg['set']], 'ShadowImages')
        self.ann_dir = os.path.join(rootdir, setmap_dict[dataset_cfg['set']], 'ShadowMasks')
        # obatin imageids
        self.imageids = []
        for line in open(os.path.join(rootdir, dataset_cfg['set']+'.txt'), 'r').readlines():
            if line.strip(): self.imageids.append(line.strip())
        self.imageids = [str(_id) for _id in self.imageids]
    '''getitem'''
    def __getitem__(self, index):
        # imageid
        imageid = self.imageids[index % len(self.imageids)]
        # read sample_meta
        imagepath = os.path.join(self.image_dir, f'{imageid}{self.image_ext}')
        annpath = os.path.join(self.ann_dir, f'{imageid}{self.ann_ext}')
        sample_meta = self.read(imagepath, annpath)
        sample_meta['seg_target'][sample_meta['seg_target'] > 0] = 1.
        # add image id
        sample_meta.update({'id': imageid})
        # synctransforms
        sample_meta = self.synctransforms(sample_meta)
        # return
        return sample_meta