'''
Function:
    Base class for loadding dataset
Author:
    Zhenchao Jin
'''
import cv2
import torch
import numpy as np
import scipy.io as sio
from PIL import Image
from chainercv.evaluations import eval_semantic_segmentation
from .pipelines import Evaluation, Resize, RandomCrop, RandomFlip, PhotoMetricDistortion, RandomRotation, Padding, ToTensor, Normalize, Compose


'''BaseDataset'''
class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, mode, logger_handle, dataset_cfg):
        assert mode in ['TRAIN', 'TEST']
        self.mode = mode
        self.logger_handle = logger_handle
        self.dataset_cfg = dataset_cfg
        self.transforms = Compose(self.constructtransforms(self.dataset_cfg['aug_opts']))
    '''pull item'''
    def __getitem__(self, index):
        raise NotImplementedError('not be implemented')
    '''length'''
    def __len__(self):
        raise NotImplementedError('not be implemented')
    '''sync transform'''
    def synctransform(self, sample, transform_type):
        assert hasattr(self, 'transforms') and self.transforms, 'undefined transforms'
        assert transform_type in ['all', 'only_totensor_normalize_pad', 'without_totensor_normalize_pad']
        sample = self.transforms(sample, transform_type)
        return sample
    '''read sample'''
    def read(self, imagepath, annpath, with_ann=True):
        assert self.mode in ['TRAIN', 'TEST']
        # read image
        image = cv2.imread(imagepath)
        # read annotation
        if annpath.endswith('.png'):
            if self.dataset_cfg['type'] in ['vspw']:
                segmentation = np.array(Image.open(annpath)) if with_ann else np.zeros((image.shape[0], image.shape[1]))
            else:
                segmentation = cv2.imread(annpath, cv2.IMREAD_GRAYSCALE) if with_ann else np.zeros((image.shape[0], image.shape[1]))
        elif annpath.endswith('.mat'):
            segmentation = sio.loadmat(annpath)
            if self.dataset_cfg['type'] in ['cocostuff10k']:
                segmentation = segmentation['S']
        else:
            raise NotImplementedError('Unsupport data type of %s' % annpath.split('.')[-1])
        if with_ann and hasattr(self, 'clsid2label'):
            for key, value in self.clsid2label.items():
                segmentation[segmentation == key] = value
        # edge placeholder
        edge = np.zeros((32, 32))
        # return sample
        sample = {
            'image': image, 
            'segmentation': segmentation.copy(), 
            'edge': edge, 
            'width': image.shape[1], 
            'height': image.shape[0]
        }
        if self.mode == 'TEST': sample.update({'groundtruth': segmentation.copy()})
        return sample
    '''construct the transforms'''
    def constructtransforms(self, aug_opts):
        # obtain the transforms
        transforms = []
        supported_transforms = {
            'Resize': Resize,
            'RandomCrop': RandomCrop,
            'RandomFlip': RandomFlip,
            'PhotoMetricDistortion': PhotoMetricDistortion,
            'RandomRotation': RandomRotation,
            'Padding': Padding,
            'ToTensor': ToTensor,
            'Normalize': Normalize
        }
        for aug_opt in aug_opts:
            key, value = aug_opt
            assert key in supported_transforms, 'unsupport transform %s' % key
            transforms.append(supported_transforms[key](**value))
        # return the transforms
        return transforms
    '''evaluate the predictions'''
    def evaluate(self, predictions, groundtruths, metric_list=['iou', 'miou'], num_classes=None, ignore_index=-1, nan_to_num=None, beta=1.0):
        eval_client = None
        result = eval_semantic_segmentation(predictions, groundtruths)
        result_selected = {}
        for metric in metric_list:
            if metric in result:
                result_selected[metric] = result[metric]
            else:
                if eval_client is None:
                    eval_client = Evaluation(predictions, groundtruths, num_classes, ignore_index, nan_to_num, beta)
                assert metric in eval_client.all_metric_results, 'unsupport %s as the metric' % metric
                result_selected[metric] = eval_client.all_metric_results[metric]
        if 'iou' in result_selected:
            iou_list = result_selected['iou']
            iou_dict = {}
            for idx, item in enumerate(iou_list):
                iou_dict[self.classnames[idx]] = item
            result_selected['iou'] = iou_dict
        if 'dice' in result_selected:
            dice_list = result_selected['dice']
            dice_dict = {}
            for idx, item in enumerate(dice_list):
                dice_dict[self.classnames[idx]] = item
            result_selected['dice'] = dice_dict
        return result_selected
    '''generate edge'''
    def generateedge(self, segmentation, edge_width=3, ignore_index=255):
        h, w = segmentation.shape
        edge = np.zeros(segmentation.shape)
        # right
        edge_right = edge[1: h, :]
        edge_right[(segmentation[1: h, :] != segmentation[:h-1, :]) & (segmentation[1: h, :] != ignore_index) & (segmentation[:h-1, :] != ignore_index)] = 1
        # up
        edge_up = edge[:, :w-1]
        edge_up[(segmentation[:, :w-1] != segmentation[:, 1: w]) & (segmentation[:, :w-1] != ignore_index) & (segmentation[:, 1: w] != ignore_index)] = 1
        # upright
        edge_upright = edge[:h-1, :w-1]
        edge_upright[(segmentation[:h-1, :w-1] != segmentation[1: h, 1: w]) & (segmentation[:h-1, :w-1] != ignore_index) & (segmentation[1: h, 1: w] != ignore_index)] = 1
        # bottomright
        edge_bottomright = edge[:h-1, 1: w]
        edge_bottomright[(segmentation[:h-1, 1: w] != segmentation[1: h, :w-1]) & (segmentation[: h-1, 1: w] != ignore_index) & (segmentation[1: h, :w-1] != ignore_index)] = 1
        # return
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (edge_width, edge_width))
        edge = cv2.dilate(edge, kernel)
        return edge