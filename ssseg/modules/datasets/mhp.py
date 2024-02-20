'''
Function:
    Implementation of MHPv1Dataset and MHPv2Dataset
Author:
    Zhenchao Jin
'''
import os
import cv2
import glob
import numpy as np
import pandas as pd
from .base import BaseDataset


'''MHPv1Dataset'''
class MHPv1Dataset(BaseDataset):
    num_classes = 19
    classnames = [
        '__background__', 'hat', 'hair', 'sunglasses', 'upper clothes', 'skirt', 'pants',
        'dress', 'belt', 'left shoe', 'right shoe', 'face', 'left leg', 'right leg',
        'left arm', 'right arm', 'bag', 'scarf', 'torso skin'
    ]
    palette = BaseDataset.randompalette(num_classes)
    assert num_classes == len(classnames) and num_classes == len(palette)
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(MHPv1Dataset, self).__init__(mode=mode, logger_handle=logger_handle, dataset_cfg=dataset_cfg)
        # obtain the dirs
        rootdir = dataset_cfg['rootdir']
        self.image_dir = os.path.join(rootdir, 'images')
        self.ann_dir = os.path.join(rootdir, 'annotations')
        # obatin imageids
        df = pd.read_csv(os.path.join(rootdir, dataset_cfg['set']+'_list.txt'), names=['imageids'])
        self.imageids = df['imageids'].values
        self.imageids = [str(_id) for _id in self.imageids]
    '''getitem'''
    def __getitem__(self, index):
        # imageid
        imageid = self.imageids[index % len(self.imageids)]
        # read sample_meta
        imagepath = os.path.join(self.image_dir, imageid)
        annpath = os.path.join(self.ann_dir, imageid.replace('.jpg', '_*'))
        image = cv2.imread(imagepath)
        seg_target = np.zeros((image.shape[0], image.shape[1]))
        for path in glob.glob(annpath):
            seg_per_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            seg_target[seg_per_image != 0] = seg_per_image[seg_per_image != 0]
        sample_meta = {
            'image': image, 'seg_target': seg_target, 'width': image.shape[1], 'height': image.shape[0],
        }
        # add image id
        sample_meta.update({'id': imageid})
        # synctransforms
        sample_meta = self.synctransforms(sample_meta)
        # return
        return sample_meta


'''MHPv2Dataset'''
class MHPv2Dataset(BaseDataset):
    num_classes = 59
    classnames = [
        '__background__', 'cap/hat', 'helmet', 'face', 'hair', 'left-arm', 'right-arm', 'left-hand', 'right-hand',
        'protector', 'bikini/bra', 'jacket/windbreaker/hoodie', 't-shirt', 'polo-shirt', 'sweater', 'sin-glet',
        'torso-skin', 'pants', 'shorts/swim-shorts', 'skirt', 'stock-ings', 'socks', 'left-boot', 'right-boot',
        'left-shoe', 'right-shoe', 'left-highheel', 'right-highheel', 'left-sandal', 'right-sandal', 'left-leg',
        'right-leg', 'left-foot', 'right-foot', 'coat', 'dress', 'robe', 'jumpsuits', 'other-full-body-clothes',
        'headwear', 'backpack', 'ball', 'bats', 'belt', 'bottle', 'carrybag', 'cases', 'sunglasses', 'eyewear',
        'gloves', 'scarf', 'umbrella', 'wallet/purse', 'watch', 'wristband', 'tie', 'other-accessaries',
        'other-upper-body-clothes', 'other-lower-body-clothes'
    ]
    palette = BaseDataset.randompalette(num_classes)
    assert num_classes == len(classnames) and num_classes == len(palette)
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(MHPv2Dataset, self).__init__(mode=mode, logger_handle=logger_handle, dataset_cfg=dataset_cfg)
        # obtain the dirs
        rootdir = dataset_cfg['rootdir']
        self.image_dir = os.path.join(rootdir, dataset_cfg['set'], 'images')
        self.ann_dir = os.path.join(rootdir, dataset_cfg['set'], 'parsing_annos')
        # obatin imageids
        df = pd.read_csv(os.path.join(rootdir, 'list', dataset_cfg['set']+'.txt'), names=['imageids'])
        self.imageids = df['imageids'].values
        self.imageids = [str(_id) for _id in self.imageids]
    '''getitem'''
    def __getitem__(self, index):
        # imageid
        imageid = self.imageids[index % len(self.imageids)]
        # read sample_meta
        imagepath = os.path.join(self.image_dir, f'{imageid}{self.image_ext}')
        annpath = os.path.join(self.ann_dir, imageid + '_*')
        image = cv2.imread(imagepath)
        seg_target = np.zeros((image.shape[0], image.shape[1]))
        for path in glob.glob(annpath):
            seg_per_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            seg_target[seg_per_image != 0] = seg_per_image[seg_per_image != 0]
        sample_meta = {
            'image': image, 'seg_target': seg_target, 'width': image.shape[1], 'height': image.shape[0],
        }
        # add image id
        sample_meta.update({'id': imageid})
        # synctransforms
        sample_meta = self.synctransforms(sample_meta)
        # return
        return sample_meta