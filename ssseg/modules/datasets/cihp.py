'''
Function:
    Load the CIHP dataset
Author:
    Zhenchao Jin
'''
import os
import pandas as pd
from .base import BaseDataset


'''CIHPDataset'''
class CIHPDataset(BaseDataset):
    num_classes = 20
    classnames = [
        '__background__', 'hat', 'hair', 'glove', 'sunglasses', 'upperclothes', 'dress', 
        'coat', 'socks', 'pants', 'torsoSkin', 'scarf', 'skirt', 'face', 
        'leftArm', 'rightArm', 'leftLeg', 'rightLeg', 'leftShoe', 'rightShoe'
    ]
    assert num_classes == len(classnames)
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(CIHPDataset, self).__init__(mode, logger_handle, dataset_cfg)
        # obtain the dirs
        rootdir = dataset_cfg['rootdir']
        setmap_dict = {'train': 'Training', 'val': 'Validation', 'test': 'Testing'}
        self.image_dir = os.path.join(rootdir, f"{setmap_dict[dataset_cfg['set']]}/Images")
        self.ann_dir = os.path.join(rootdir, f"{setmap_dict[dataset_cfg['set']]}/Category_ids")
        # obatin imageids
        df = pd.read_csv(os.path.join(rootdir, setmap_dict[dataset_cfg['set']], dataset_cfg['set']+'_id.txt'), names=['imageids'])
        self.imageids = df['imageids'].values
        self.imageids = [str(_id).zfill(7) for _id in self.imageids]
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