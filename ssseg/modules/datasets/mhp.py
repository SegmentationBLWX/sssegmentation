'''
Function:
    Load the MHPv1 and MHPv2 dataset
Author:
    Zhenchao Jin
'''
import os
import glob
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
    assert num_classes == len(classnames)
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(MHPv1Dataset, self).__init__(mode, logger_handle, dataset_cfg)
        # obtain the dirs
        rootdir = dataset_cfg['rootdir']
        self.image_dir = os.path.join(rootdir, 'images')
        self.ann_dir = os.path.join(rootdir, 'annotations')
        # obatin imageids
        df = pd.read_csv(os.path.join(rootdir, dataset_cfg['set']+'_list.txt'), names=['imageids'])
        self.imageids = df['imageids'].values
        self.imageids = [str(_id) for _id in self.imageids]
    '''pull item'''
    def __getitem__(self, index):
        imageid = self.imageids[index]
        imagepath = os.path.join(self.image_dir, imageid)
        annpath = os.path.join(self.ann_dir, imageid.replace('.jpg', '_*'))
        sample = self.read(imagepath, annpath, False)
        if self.dataset_cfg.get('with_ann', True):
            segmentation = sample['segmentation']
            for path in glob.glob(annpath):
                seg_per_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                segmentation[seg_per_image != 0] = seg_per_image[seg_per_image != 0]
            if 'segmentation' in sample: sample['segmentation'] = segmentation
            if 'groundtruth' in sample: sample['groundtruth'] = segmentation
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
    assert num_classes == len(classnames)
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(MHPv2Dataset, self).__init__(mode, logger_handle, dataset_cfg)
        # obtain the dirs
        rootdir = dataset_cfg['rootdir']
        self.image_dir = os.path.join(rootdir, dataset_cfg['set'], 'images')
        self.ann_dir = os.path.join(rootdir, dataset_cfg['set'], 'parsing_annos')
        # obatin imageids
        df = pd.read_csv(os.path.join(rootdir, 'list', dataset_cfg['set']+'.txt'), names=['imageids'])
        self.imageids = df['imageids'].values
        self.imageids = [str(_id) for _id in self.imageids]
    '''pull item'''
    def __getitem__(self, index):
        imageid = self.imageids[index]
        imagepath = os.path.join(self.image_dir, imageid+'.jpg')
        annpath = os.path.join(self.ann_dir, imageid + '_*')
        sample = self.read(imagepath, annpath, False)
        if self.dataset_cfg.get('with_ann', True):
            segmentation = sample['segmentation']
            for path in glob.glob(annpath):
                seg_per_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                segmentation[seg_per_image != 0] = seg_per_image[seg_per_image != 0]
            if 'segmentation' in sample: sample['segmentation'] = segmentation
            if 'groundtruth' in sample: sample['groundtruth'] = segmentation
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