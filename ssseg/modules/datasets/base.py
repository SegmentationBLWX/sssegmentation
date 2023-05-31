'''
Function:
    Implementation of BaseDataset
Author:
    Zhenchao Jin
'''
import os
import cv2
import torch
import numpy as np
import scipy.io as sio
from PIL import Image
from chainercv.evaluations import eval_semantic_segmentation
from .pipelines import (
    Evaluation, Resize, RandomCrop, RandomFlip, PhotoMetricDistortion,
    RandomRotation, Padding, ToTensor, Normalize, Compose, EdgeExtractor
)


'''BaseDataset'''
class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, mode, logger_handle, dataset_cfg):
        # assert
        assert mode in ['TRAIN', 'TEST']
        # set attributes
        self.mode = mode
        self.ann_ext = '.png'
        self.image_ext = '.jpg'
        self.dataset_cfg = dataset_cfg
        self.logger_handle = logger_handle
        self.repeat_times = dataset_cfg.get('repeat_times', 1)
        self.transforms = self.constructtransforms(self.dataset_cfg['data_pipelines'])
    '''getitem'''
    def __getitem__(self, index):
        # imageid
        imageid = self.imageids[index % len(self.imageids)]
        # read sample_meta
        imagepath = os.path.join(self.image_dir, f'{imageid}{self.image_ext}')
        annpath = os.path.join(self.ann_dir, f'{imageid}{self.ann_ext}')
        sample_meta = self.read(imagepath, annpath)
        # add image id
        sample_meta.update({'id': imageid})
        # synctransforms
        sample_meta = self.synctransforms(sample_meta)
        # return
        return sample_meta
    '''len'''
    def __len__(self):
        return len(self.imageids) * self.repeat_times
    '''read sample_meta'''
    def read(self, imagepath, annpath):
        # read image
        image = cv2.imread(imagepath)
        # read annotation
        assert annpath.split('.')[-1] in ['png', 'mat']
        if not os.path.exists(annpath):
            seg_target = np.zeros((image.shape[0], image.shape[1]))
            self.logger_handle.warning(f'seg_target from {annpath} is missed')
        elif annpath.endswith('.png'):
            if self.dataset_cfg['type'] in ['VSPWDataset']:
                seg_target = np.array(Image.open(annpath))
            else:
                seg_target = cv2.imread(annpath, cv2.IMREAD_GRAYSCALE)
        elif annpath.endswith('.mat'):
            seg_target = sio.loadmat(annpath)
            if self.dataset_cfg['type'] in ['COCOStuff10kDataset']:
                seg_target = seg_target['S']
        else:
            seg_target = np.zeros((image.shape[0], image.shape[1]))
            self.logger_handle.warning(f'seg_target from {annpath} is not supported to read')
        # auto transform seg_target to train labels
        if hasattr(self, 'clsid2label'):
            for key, value in self.clsid2label.items():
                seg_target[seg_target == key] = value
        # construct sample_meta
        sample_meta = {
            'image': image, 'seg_target': seg_target, 'width': image.shape[1], 'height': image.shape[0],
        }
        # return
        return sample_meta
    '''evaluate'''
    def evaluate(self, seg_preds, seg_targets, metric_list=['iou', 'miou'], num_classes=None, ignore_index=-1, nan_to_num=None, beta=1.0):
        # basic evaluation
        result = eval_semantic_segmentation(seg_preds, seg_targets)
        # selected result
        selected_result, eval_client = {}, None
        for metric in metric_list:
            if metric in result:
                selected_result[metric] = result[metric]
            else:
                if eval_client is None: eval_client = Evaluation(seg_preds, seg_targets, num_classes, ignore_index, nan_to_num, beta)
                assert metric in eval_client.all_metric_results
                selected_result[metric] = eval_client.all_metric_results[metric]
        # insert class names for iou and dice
        if 'iou' in selected_result:
            iou_list, iou_dict = selected_result['iou'], {}
            for idx, item in enumerate(iou_list):
                iou_dict[self.classnames[idx]] = item
            selected_result['iou'] = iou_dict
        if 'dice' in selected_result:
            dice_list, dice_dict = selected_result['dice'], {}
            for idx, item in enumerate(dice_list):
                dice_dict[self.classnames[idx]] = item
            selected_result['dice'] = dice_dict
        # return
        return selected_result      
    '''constructtransforms'''
    def constructtransforms(self, data_pipelines):
        # supported transforms
        supported_transforms = {
            'Resize': Resize, 'RandomCrop': RandomCrop, 'RandomFlip': RandomFlip, 'RandomRotation': RandomRotation, 'EdgeExtractor': EdgeExtractor,
            'PhotoMetricDistortion': PhotoMetricDistortion, 'Padding': Padding, 'ToTensor': ToTensor, 'Normalize': Normalize,
        }
        # build transforms
        transforms = []
        for data_pipeline in data_pipelines:
            assert len(data_pipeline) == 2
            transforms.append(supported_transforms[data_pipeline[0]](**data_pipeline[1]))
        transforms = Compose(transforms)
        # return
        return transforms
    '''synctransforms'''
    def synctransforms(self, sample_meta):
        if self.mode == 'TEST':
            seg_target = sample_meta.pop('seg_target')
        sample_meta = self.transforms(sample_meta)
        if self.mode == 'TEST':
            sample_meta['seg_target'] = seg_target
        return sample_meta
    '''randompalette'''
    @staticmethod
    def randompalette(num_classes):
        palette = [0] * (num_classes * 3)
        for j in range(0, num_classes):
            i, lab = 0, j
            palette[j * 3 + 0], palette[j * 3 + 1], palette[j * 3 + 2] = 0, 0, 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        palette = np.array(palette).reshape(-1, 3)
        palette = palette.tolist()
        return palette