'''
Function:
    Implementation of CIHPDataset
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
    palette = [
        (0, 0, 0), (128, 0, 0), (255, 0, 0), (0, 85, 0), (170, 0, 51), (255, 85, 0), (0, 0, 85),
        (0, 119, 221), (85, 85, 0), (0, 85, 85), (85, 51, 0), (52, 86, 128), (0, 128, 0), (0, 0, 255), 
        (51, 170, 221), (0, 255, 255), (85, 255, 170), (170, 255, 85), (255, 255, 0), (255, 170, 0)
    ]
    assert num_classes == len(classnames) and num_classes == len(palette)
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(CIHPDataset, self).__init__(mode=mode, logger_handle=logger_handle, dataset_cfg=dataset_cfg)
        # obtain the dirs
        rootdir = dataset_cfg['rootdir']
        setmap_dict = {'train': 'Training', 'val': 'Validation', 'test': 'Testing'}
        self.image_dir = os.path.join(rootdir, f"{setmap_dict[dataset_cfg['set']]}/Images")
        self.ann_dir = os.path.join(rootdir, f"{setmap_dict[dataset_cfg['set']]}/Category_ids")
        # obatin imageids
        df = pd.read_csv(os.path.join(rootdir, setmap_dict[dataset_cfg['set']], dataset_cfg['set']+'_id.txt'), names=['imageids'])
        self.imageids = df['imageids'].values
        self.imageids = [str(_id).zfill(7) for _id in self.imageids]