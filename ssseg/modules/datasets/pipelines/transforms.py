'''
Function:
    Implementation of Transforms
Author:
    Zhenchao Jin
'''
import cv2
import copy
import torch
import numbers
import collections
import numpy as np
import torch.nn.functional as F
from PIL import ImageFilter, Image
from ...utils import BaseModuleBuilder


'''_INTERPOLATION_CV2_CONVERTOR'''
_INTERPOLATION_CV2_CONVERTOR = {
    'area': cv2.INTER_AREA, 'bicubic': cv2.INTER_CUBIC, 'bilinear': cv2.INTER_LINEAR, 
    'nearest': cv2.INTER_NEAREST, 'lanczos': cv2.INTER_LANCZOS4,
}


'''Resize'''
class Resize(object):
    def __init__(self, output_size, keep_ratio=True, min_size=None, scale_range=(0.5, 2.0), image_interpolation='bilinear', seg_target_interpolation='nearest', img2aug_pos_mapper_interpolation='nearest'):
        # assert
        assert isinstance(scale_range, collections.abc.Sequence) or scale_range is None
        assert isinstance(output_size, int) or (isinstance(output_size, collections.abc.Sequence) and len(output_size) == 2)
        # set attributes
        self.min_size = min_size
        self.keep_ratio = keep_ratio
        self.scale_range = scale_range
        self.image_interpolation = _INTERPOLATION_CV2_CONVERTOR[image_interpolation]
        self.seg_target_interpolation = _INTERPOLATION_CV2_CONVERTOR[seg_target_interpolation]
        self.img2aug_pos_mapper_interpolation = _INTERPOLATION_CV2_CONVERTOR[img2aug_pos_mapper_interpolation]
        self.output_size = (output_size, output_size) if isinstance(output_size, int) else output_size
    '''call'''
    def __call__(self, sample_meta):
        # calculate output_size
        if self.scale_range is not None:
            rand_scale = np.random.random_sample() * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
            output_size = int(self.output_size[0] * rand_scale), int(self.output_size[1] * rand_scale)
        else:
            output_size = self.output_size[0], self.output_size[1]
        # deal with keep_ratio
        if self.keep_ratio:
            scale_factor = min(max(output_size) / max(sample_meta['image'].shape[:2]), min(output_size) / min(sample_meta['image'].shape[:2]))
            dsize = int(sample_meta['image'].shape[1] * scale_factor + 0.5), int(sample_meta['image'].shape[0] * scale_factor + 0.5)
            if self.min_size is not None and min(dsize) < self.min_size:
                scale_factor = self.min_size / min(sample_meta['image'].shape[:2])
                dsize = int(sample_meta['image'].shape[1] * scale_factor + 0.5), int(sample_meta['image'].shape[0] * scale_factor + 0.5)        
        else:
            if sample_meta['image'].shape[0] > sample_meta['image'].shape[1]:
                dsize = min(output_size), max(output_size)
            else:
                dsize = max(output_size), min(output_size)
        # resize
        sample_meta = self.resize('image', sample_meta, dsize, self.image_interpolation)
        sample_meta = self.resize('seg_target', sample_meta, dsize, self.seg_target_interpolation)
        sample_meta = self.resize('img2aug_pos_mapper', sample_meta, dsize, self.img2aug_pos_mapper_interpolation)
        # return
        return sample_meta
    '''resize'''
    @staticmethod
    def resize(key, sample_meta, dsize, interpolation):
        if key not in sample_meta: return sample_meta
        sample_meta[key] = cv2.resize(sample_meta[key], dsize=dsize, interpolation=interpolation)
        return sample_meta


'''RandomCrop'''
class RandomCrop(object):
    def __init__(self, crop_size, ignore_index=255, one_category_max_ratio=0.75):
        # assert
        assert isinstance(crop_size, int) or (isinstance(crop_size, collections.abc.Sequence) and len(crop_size) == 2)
        # set attributes
        self.ignore_index = ignore_index
        self.one_category_max_ratio = one_category_max_ratio
        self.crop_size = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size
    '''call'''
    def __call__(self, sample_meta):
        # avoid the cropped image is filled by only one category
        for _ in range(10):
            # --parse
            image, seg_target = sample_meta['image'].copy(), sample_meta['seg_target'].copy()
            h_ori, w_ori = image.shape[:2]
            h_out, w_out = min(self.crop_size[0], h_ori), min(self.crop_size[1], w_ori)
            # --random crop
            top, left = np.random.randint(0, h_ori - h_out + 1), np.random.randint(0, w_ori - w_out + 1)
            image = self.crop(image, top, left, h_out, w_out)
            seg_target = self.crop(seg_target, top, left, h_out, w_out)
            if 'img2aug_pos_mapper' in sample_meta:
                img2aug_pos_mapper = sample_meta['img2aug_pos_mapper'].copy()
                img2aug_pos_mapper = self.crop(img2aug_pos_mapper, top, left, h_out, w_out)
            # --judge
            labels, counts = np.unique(seg_target, return_counts=True)
            counts = counts[labels != self.ignore_index]
            if len(counts) > 1 and np.max(counts) / np.sum(counts) < self.one_category_max_ratio:
                break
        # update
        if len(counts) == 0: return sample_meta
        sample_meta['image'], sample_meta['seg_target'] = image, seg_target
        if 'img2aug_pos_mapper' in sample_meta:
            sample_meta['img2aug_pos_mapper'] = img2aug_pos_mapper
        # return
        return sample_meta
    '''crop'''
    @staticmethod
    def crop(image, top, left, h_out, w_out):
        image = image[top: top + h_out, left: left + w_out]
        return image


'''ResizeShortestEdge'''
class ResizeShortestEdge(object):
    def __init__(self, short_edge_length, max_size, **resize_kwargs):
        # set attributes
        self.max_size = max_size
        self.resize_kwargs = resize_kwargs
        self.short_edge_length = short_edge_length
    '''call'''
    def __call__(self, sample_meta):
        return self.resize(sample_meta, self.getoutputshape, **self.resize_kwargs)
    '''getoutputshape'''
    def getoutputshape(self, image):
        h, w = image.shape[:2]
        # calculate new size
        short_edge_length = self.short_edge_length
        if isinstance(short_edge_length, int):
            size = short_edge_length * 1.0
        elif isinstance(short_edge_length, collections.abc.Sequence):
            size = min(short_edge_length) * 1.0
        scale = size / min(h, w)
        if h < w: new_h, new_w = size, scale * w
        else: new_h, new_w = scale * h, size
        # clip if > max_size
        if max(new_h, new_w) > self.max_size:
            scale = self.max_size * 1.0 / max(new_h, new_w)
            new_h *= scale
            new_w *= scale
        new_h = int(new_h + 0.5)
        new_w = int(new_w + 0.5)
        # return new size
        return (new_w, new_h)
    '''resize'''
    @staticmethod
    def resize(sample_meta, getoutputshape_func, **resize_kwargs):
        resize_kwargs.update({'output_size': getoutputshape_func(sample_meta['image']), 'keep_ratio': True, 'scale_range': None})
        resize_transform = Resize(**resize_kwargs)
        return resize_transform(sample_meta)


'''RandomShortestEdgeResize'''
class RandomShortestEdgeResize(object):
    def __init__(self, short_edge_range, max_size, **resize_kwargs):
        # assert
        assert isinstance(short_edge_range, collections.abc.Sequence) and len(short_edge_range) == 2
        # set attributes
        self.max_size = max_size
        self.resize_kwargs = resize_kwargs
        self.short_edge_range = short_edge_range
    '''call'''
    def __call__(self, sample_meta):
        short_edge_length = np.random.randint(*self.short_edge_range)
        sample_meta = ResizeShortestEdge(short_edge_length, self.max_size, **self.resize_kwargs)(sample_meta)
        return sample_meta


'''RandomChoiceResize'''
class RandomChoiceResize(object):
    def __init__(self, scales, resize_type='Resize', **resize_kwargs):
        # set attributes
        if isinstance(scales, collections.abc.Sequence): self.scales = scales
        else: self.scales = [scales]
        self.resize_type = resize_type
        self.resize_kwargs = resize_kwargs
        self.resize_type_key_convertor = {
            'Resize': 'output_size', 'ResizeShortestEdge': 'short_edge_length'
        }
        assert resize_type in self.resize_type_key_convertor, f'unsupport resize_type {resize_type}'
        # fetch resize transform
        self.resize_transform = {
            'Resize': Resize, 'ResizeShortestEdge': ResizeShortestEdge
        }[resize_type]
    '''call'''
    def __call__(self, sample_meta):
        # random select scale
        target_scale, target_scale_idx = self.randomselect()
        # obtain kwargs
        kwargs = copy.deepcopy(self.resize_kwargs)
        kwargs[self.resize_type_key_convertor[self.resize_type]] = target_scale
        # obtain resize_transform
        resize_transform = self.resize_transform(**kwargs)
        # resize
        sample_meta = self.resize(sample_meta, resize_transform)
        # return
        return sample_meta
    '''randomselect'''
    def randomselect(self):
        target_scale_idx = np.random.randint(len(self.scales))
        target_scale = self.scales[target_scale_idx]
        return target_scale, target_scale_idx
    '''resize'''
    @staticmethod
    def resize(sample_meta, resize_transform):
        return resize_transform(sample_meta)


'''AdjustGamma'''
class AdjustGamma(object):
    def __init__(self, gamma=1.0):
        # assert
        assert gamma > 0.0
        assert isinstance(gamma, float) or isinstance(gamma, int)
        # set attributes
        self.gamma = gamma
        inv_gamma = 1.0 / gamma
        self.table = np.array([(i / 255.0)**inv_gamma * 255 for i in np.arange(256)]).astype('uint8')
        assert self.table.shape == (256, )
    '''call'''
    def __call__(self, sample_meta):
        # assert
        assert isinstance(sample_meta['image'], np.ndarray)
        assert 0 <= np.min(sample_meta['image']) and np.max(sample_meta['image']) <= 255
        # perform lut transform
        sample_meta = self.lut('image', sample_meta, self.table)
        # return
        return sample_meta
    '''lut'''
    @staticmethod
    def lut(key, sample_meta, lut_table):
        if key not in sample_meta: return sample_meta
        sample_meta[key] = cv2.LUT(np.array(sample_meta[key], dtype=np.uint8), lut_table)
        return sample_meta


'''Rerange'''
class Rerange(object):
    def __init__(self, min_value=0, max_value=255):
        # assert
        assert min_value < max_value
        assert isinstance(min_value, (float, int))
        assert isinstance(max_value, (float, int))
        # set attributes
        self.min_value = min_value
        self.max_value = max_value
    '''call'''
    def __call__(self, sample_meta):
        sample_meta = self.rerange('image', sample_meta, self.min_value, self.max_value)
        return sample_meta
    '''rerange'''
    @staticmethod
    def rerange(key, sample_meta, min_value, max_value):
        if key not in sample_meta: return sample_meta
        # max and min value in sample_meta[key]
        key_min_value = np.min(sample_meta[key])
        key_max_value = np.max(sample_meta[key])
        assert key_min_value < key_max_value
        # rerange to [0, 1]
        sample_meta[key] = (sample_meta[key] - key_min_value) / (key_max_value - key_min_value)
        # rerange to [min_value, max_value]
        sample_meta[key] = sample_meta[key] * (max_value - min_value) + min_value
        # return
        return sample_meta


'''CLAHE'''
class CLAHE(object):
    def __init__(self, clip_limit=40.0, tile_grid_size=(8, 8)):
        # assert
        assert isinstance(clip_limit, (float, int))
        assert isinstance(tile_grid_size, collections.abc.Sequence)
        assert len(tile_grid_size) == 2
        # set attribute
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
    '''call'''
    def __call__(self, sample_meta):
        sample_meta = self.clahe('image', sample_meta, self.clip_limit, self.tile_grid_size)
        return sample_meta
    '''clahe'''
    @staticmethod
    def clahe(key, sample_meta, clip_limit, tile_grid_size):
        if key not in sample_meta: return sample_meta
        for i in range(sample_meta[key].shape[2]):
            clahe = cv2.createCLAHE(clip_limit, tile_grid_size)
            sample_meta[key][:, :, i] = clahe.apply(np.array(sample_meta[key][:, :, i], dtype=np.uint8))
        return sample_meta


'''RGB2Gray'''
class RGB2Gray(object):
    def __init__(self, out_channels=None, weights=(0.299, 0.587, 0.114)):
        # assert
        assert isinstance(weights, collections.abc.Sequence)
        assert out_channels is None or out_channels > 0
        for item in weights: assert isinstance(item, (float, int))
        # set attributes
        self.weights = weights
        self.out_channels = out_channels
    '''call'''
    def __call__(self, sample_meta):
        sample_meta = self.rgb2gray('image', sample_meta, self.weights, self.out_channels)
        return sample_meta
    '''rgb2gray'''
    @staticmethod
    def rgb2gray(key, sample_meta, weights, out_channels):
        if key not in sample_meta: return sample_meta
        # assert
        assert len(sample_meta[key].shape) == 3
        assert sample_meta[key].shape[2] == len(weights)
        # apply
        weights = np.array(weights).reshape((1, 1, -1))
        sample_meta[key] = (sample_meta[key] * weights).sum(2, keepdims=True)
        if out_channels is None:
            sample_meta[key] = sample_meta[key].repeat(weights.shape[2], axis=2)
        else:
            sample_meta[key] = sample_meta[key].repeat(out_channels, axis=2)
        # return
        return sample_meta


'''RandomCutOut'''
class RandomCutOut(object):
    def __init__(self, prob, n_holes, cutout_shape=None, cutout_ratio=None, image_fill_value=(0, 0, 0), seg_target_fill_value=255, img2aug_pos_mapper_fill_value=-1):
        # assert
        assert 0 <= prob and prob <= 1
        assert (cutout_shape is None) ^ (cutout_ratio is None), 'either cutout_shape or cutout_ratio should be specified'
        assert (isinstance(cutout_shape, collections.abc.Sequence) or isinstance(cutout_ratio, collections.abc.Sequence))
        if isinstance(n_holes, collections.abc.Sequence):
            assert len(n_holes) == 2 and 0 <= n_holes[0] < n_holes[1]
        else:
            n_holes = (n_holes, n_holes)
        if seg_target_fill_value is not None:
            assert (isinstance(seg_target_fill_value, int) and 0 <= seg_target_fill_value and seg_target_fill_value <= 255)
        # set attributes
        self.prob = prob
        self.n_holes = n_holes
        self.image_fill_value = image_fill_value
        self.seg_target_fill_value = seg_target_fill_value
        self.img2aug_pos_mapper_fill_value = img2aug_pos_mapper_fill_value
        self.with_ratio = cutout_ratio is not None
        self.candidates = cutout_ratio if self.with_ratio else cutout_shape
        if not isinstance(self.candidates, collections.abc.Sequence): self.candidates = [self.candidates]
    '''call'''
    def __call__(self, sample_meta):
        cutout, n_holes, x1_lst, y1_lst, index_lst = self.generatepatches(sample_meta['image'])
        if cutout:
            h, w, c = sample_meta['image'].shape
            for i in range(n_holes):
                x1, y1, index = x1_lst[i], y1_lst[i], index_lst[i]
                if not self.with_ratio:
                    cutout_w, cutout_h = self.candidates[index]
                else:
                    cutout_w = int(self.candidates[index][0] * w)
                    cutout_h = int(self.candidates[index][1] * h)
                x2 = np.clip(x1 + cutout_w, 0, w)
                y2 = np.clip(y1 + cutout_h, 0, h)
                sample_meta = self.cutout('image', sample_meta, x1, y1, x2, y2, self.image_fill_value)
                sample_meta = self.cutout('seg_target', sample_meta, x1, y1, x2, y2, self.seg_target_fill_value)
                sample_meta = self.cutout('img2aug_pos_mapper', sample_meta, x1, y1, x2, y2, self.img2aug_pos_mapper_fill_value)
        return sample_meta
    '''docutout'''
    def docutout(self):
        return np.random.rand() < self.prob
    '''generatepatches'''
    def generatepatches(self, image):
        cutout = self.docutout()
        h, w, _ = image.shape
        # obtain n_holes
        if cutout:
            n_holes = np.random.randint(self.n_holes[0], self.n_holes[1] + 1)
        else:
            n_holes = 0
        # generate patches
        x1_lst, y1_lst, index_lst = [], [], []
        for _ in range(n_holes):
            x1_lst.append(np.random.randint(0, w))
            y1_lst.append(np.random.randint(0, h))
            index_lst.append(np.random.randint(0, len(self.candidates)))
        # return
        return cutout, n_holes, x1_lst, y1_lst, index_lst
    '''cutout'''
    @staticmethod
    def cutout(key, sample_meta, x1, y1, x2, y2, fill_value):
        if key not in sample_meta: return sample_meta
        sample_meta[key][y1: y2, x1: x2, :] = fill_value
        return sample_meta


'''AlbumentationsWrapper'''
class AlbumentationsWrapper():
    def __init__(self, albu_cfg, albu_forward_args={}, transform_keys=['image', 'seg_target']):
        # assert
        assert isinstance(albu_cfg, dict)
        assert isinstance(transform_keys, collections.abc.Sequence)
        # set attributes
        self.albu_forward_args = albu_forward_args
        self.transform_keys = transform_keys
        self.transform_key_convertor = {
            'image': 'image', 'seg_target': 'mask',
        }
        # build albu transform
        import albumentations
        albu_type = albu_cfg.pop('type')
        assert hasattr(albumentations, albu_type), f'Albumentations lib unsupport {albu_type}, refer to https://github.com/albumentations-team/albumentations for more details'
        self.albu_transform = getattr(albumentations, albu_type)(**albu_cfg)
    '''call'''
    def __call__(self, sample_meta):
        albu_forward_args = copy.deepcopy(self.albu_forward_args)
        for key in self.transform_keys:
            albu_forward_args[self.transform_key_convertor[key]] = sample_meta[key]
        transformed_results = self.albu(self.albu_transform, albu_forward_args)
        for key in self.transform_keys:
            sample_meta[key] = transformed_results[self.transform_key_convertor[key]]
        return sample_meta
    '''albu'''
    @staticmethod
    def albu(albu_transform, albu_forward_args):
        transformed_results = albu_transform(**albu_forward_args)
        return transformed_results


'''RandomFlip'''
class RandomFlip(object):
    def __init__(self, flip_prob, fixed_seg_target_pairs=None):
        # assert
        assert isinstance(flip_prob, float)
        # set attributes
        self.flip_prob = flip_prob
        self.fixed_seg_target_pairs = fixed_seg_target_pairs
    '''call'''
    def __call__(self, sample_meta):
        # flip
        if np.random.rand() < self.flip_prob: return sample_meta
        sample_meta = self.flip('image', sample_meta)
        sample_meta = self.flip('seg_target', sample_meta)
        sample_meta = self.flip('img2aug_pos_mapper', sample_meta)
        # fix some seg_target pairs (used in some human parsing datasets)
        if self.fixed_seg_target_pairs:
            for (pair_a, pair_b) in self.fixed_seg_target_pairs:
                pair_a_pos = np.where(sample_meta['seg_target'] == pair_a)
                pair_b_pos = np.where(sample_meta['seg_target'] == pair_b)
                sample_meta['seg_target'][pair_a_pos[0], pair_a_pos[1]] = pair_b
                sample_meta['seg_target'][pair_b_pos[0], pair_b_pos[1]] = pair_a
        # return
        return sample_meta
    '''flip'''
    @staticmethod
    def flip(key, sample_meta):
        if key not in sample_meta: return sample_meta
        sample_meta[key] = np.flip(sample_meta[key], axis=1)
        return sample_meta


'''PhotoMetricDistortion'''
class PhotoMetricDistortion(object):
    def __init__(self, brightness_delta=32, contrast_range=(0.5, 1.5), saturation_range=(0.5, 1.5), hue_delta=18):
        # assert
        assert isinstance(contrast_range, collections.abc.Sequence) and len(contrast_range) == 2
        assert isinstance(saturation_range, collections.abc.Sequence) and len(saturation_range) == 2
        # set attributes
        self.hue_delta = hue_delta
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
    '''call'''
    def __call__(self, sample_meta):
        sample_meta['image'] = self.brightness(sample_meta['image'])
        mode = np.random.randint(2)
        if mode == 1: sample_meta['image'] = self.contrast(sample_meta['image'])
        sample_meta['image'] = self.saturation(sample_meta['image'])
        sample_meta['image'] = self.hue(sample_meta['image'])
        if mode == 0: sample_meta['image'] = self.contrast(sample_meta['image'])
        return sample_meta
    '''brightness distortion'''
    def brightness(self, image):
        if not np.random.randint(2): return image
        return self.convert(image, beta=np.random.uniform(-self.brightness_delta, self.brightness_delta))
    '''contrast distortion'''
    def contrast(self, image):
        if not np.random.randint(2): return image
        return self.convert(image, alpha=np.random.uniform(self.contrast_lower, self.contrast_upper))
    '''rgb2hsv'''
    def rgb2hsv(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    '''hsv2rgb'''
    def hsv2rgb(self, image):
        return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    '''saturation distortion'''
    def saturation(self, image):
        if not np.random.randint(2): return image
        image = self.rgb2hsv(image)
        image[..., 1] = self.convert(image[..., 1], alpha=np.random.uniform(self.saturation_lower, self.saturation_upper))
        image = self.hsv2rgb(image)
        return image
    '''hue distortion'''
    def hue(self, image):
        if not np.random.randint(2): return image
        image = self.rgb2hsv(image)
        image[..., 0] = (image[..., 0].astype(int) + np.random.randint(-self.hue_delta, self.hue_delta)) % 180
        image = self.hsv2rgb(image)
        return image
    '''multiple with alpha and add beat with clip'''
    def convert(self, image, alpha=1, beta=0):
        image = image.astype(np.float32) * alpha + beta
        image = np.clip(image, 0, 255)
        return image.astype(np.uint8)


'''RandomRotation'''
class RandomRotation(object):
    def __init__(self, rotation_prob=0.5, angle_upper=30, image_fill_value=0.0, seg_target_fill_value=255, image_interpolation='bicubic', seg_target_interpolation='nearest', img2aug_pos_mapper_fill_value=-1, img2aug_pos_mapper_interpolation='nearest'):
        # assert
        assert isinstance(rotation_prob, float)
        # set attributes
        self.rotation_prob = rotation_prob
        self.angle_upper = angle_upper
        self.image_fill_value = image_fill_value
        self.seg_target_fill_value = seg_target_fill_value
        self.img2aug_pos_mapper_fill_value = img2aug_pos_mapper_fill_value
        self.image_interpolation = _INTERPOLATION_CV2_CONVERTOR[image_interpolation]
        self.seg_target_interpolation = _INTERPOLATION_CV2_CONVERTOR[seg_target_interpolation]
        self.img2aug_pos_mapper_interpolation = _INTERPOLATION_CV2_CONVERTOR[img2aug_pos_mapper_interpolation]
    '''call'''
    def __call__(self, sample_meta):
        # prepare
        if np.random.rand() < self.rotation_prob: return sample_meta
        h_ori, w_ori = sample_meta['image'].shape[:2]
        rand_angle = np.random.randint(-self.angle_upper, self.angle_upper)
        matrix = cv2.getRotationMatrix2D(center=(w_ori / 2, h_ori / 2), angle=rand_angle, scale=1)
        # rotate
        sample_meta = self.rotate('image', sample_meta, matrix, w_ori, h_ori, self.image_interpolation, self.image_fill_value)
        sample_meta = self.rotate('seg_target', sample_meta, matrix, w_ori, h_ori, self.seg_target_interpolation, self.seg_target_fill_value)
        sample_meta = self.rotate('img2aug_pos_mapper', sample_meta, matrix, w_ori, h_ori, self.img2aug_pos_mapper_interpolation, self.img2aug_pos_mapper_fill_value)
        # return
        return sample_meta
    '''rotate'''
    @staticmethod
    def rotate(key, sample_meta, matrix, w_ori, h_ori, interpolation, fill_value):
        if key not in sample_meta: return sample_meta
        sample_meta[key] = cv2.warpAffine(sample_meta[key], matrix, (w_ori, h_ori), flags=interpolation, borderValue=fill_value)
        return sample_meta


'''RandomGaussianBlur'''
class RandomGaussianBlur(object):
    def __init__(self, prob=0.5, sigma=[0., 1.0], kernel_size=3):
        super(RandomGaussianBlur, self).__init__()
        assert (isinstance(sigma, collections.abc.Sequence) and (len(sigma) == 2)) or isinstance(sigma, numbers.Number)
        assert (isinstance(kernel_size, collections.abc.Sequence) and (len(kernel_size) == 2)) or isinstance(kernel_size, numbers.Number)
        self.prob = prob
        self.sigma = sigma if isinstance(sigma, collections.abc.Sequence) else [sigma, sigma]
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, numbers.Number) else kernel_size
    '''call'''
    def __call__(self, sample_meta):
        if np.random.rand() < self.prob: return sample_meta
        sigma = np.random.uniform(self.sigma[0], self.sigma[1])
        sample_meta = self.gaussianblur('image', sample_meta, self.kernel_size, sigma)
        return sample_meta
    '''gaussianblur'''
    @staticmethod
    def gaussianblur(key, sample_meta, kernel_size, sigma):
        if key not in sample_meta: return sample_meta
        sample_meta[key] = cv2.GaussianBlur(sample_meta[key], kernel_size, sigma)
        return sample_meta


'''PILRandomGaussianBlur'''
class PILRandomGaussianBlur(object):
    def __init__(self, prob=0.5, radius=[0., 1.0]):
        super(PILRandomGaussianBlur, self).__init__()
        assert (isinstance(radius, collections.abc.Sequence) and (len(radius) == 2)) or isinstance(radius, numbers.Number)
        self.prob = prob
        self.radius = radius if isinstance(radius, collections.abc.Sequence) else [radius, radius]
    '''call'''
    def __call__(self, sample_meta):
        if np.random.rand() < self.prob: return sample_meta
        radius = np.random.uniform(self.radius[0], self.radius[1])
        sample_meta = self.gaussianblur('image', sample_meta, radius)
        return sample_meta
    '''gaussianblur'''
    @staticmethod
    def gaussianblur(key, sample_meta, radius):
        if key not in sample_meta: return sample_meta
        # convert to pillow style
        sample_meta[key] = cv2.cvtColor(sample_meta[key], cv2.COLOR_BGR2RGB)
        sample_meta[key] = Image.fromarray(sample_meta[key])
        # perform gaussian filter
        sample_meta[key] = sample_meta[key].filter(ImageFilter.GaussianBlur(radius=radius))
        # convert return to cv2 style
        sample_meta[key] = np.array(sample_meta[key])[:, :, ::-1]
        # return
        return sample_meta


'''EdgeExtractor'''
class EdgeExtractor(object):
    def __init__(self, edge_width=3, ignore_index=255):
        # assert
        assert isinstance(edge_width, int)
        # set attributes
        self.edge_width = edge_width
        self.ignore_index = ignore_index
    '''call'''
    def __call__(self, sample_meta):
        # prepare
        seg_target = sample_meta['seg_target']
        (h, w), edge, edge_width, ignore_index = seg_target.shape, np.zeros(seg_target.shape), self.edge_width, self.ignore_index
        # right
        edge_right = edge[1: h, :]
        edge_right[(seg_target[1: h, :] != seg_target[:h-1, :]) & (seg_target[1: h, :] != ignore_index) & (seg_target[:h-1, :] != ignore_index)] = 1
        # up
        edge_up = edge[:, :w-1]
        edge_up[(seg_target[:, :w-1] != seg_target[:, 1: w]) & (seg_target[:, :w-1] != ignore_index) & (seg_target[:, 1: w] != ignore_index)] = 1
        # upright
        edge_upright = edge[:h-1, :w-1]
        edge_upright[(seg_target[:h-1, :w-1] != seg_target[1: h, 1: w]) & (seg_target[:h-1, :w-1] != ignore_index) & (seg_target[1: h, 1: w] != ignore_index)] = 1
        # bottomright
        edge_bottomright = edge[:h-1, 1: w]
        edge_bottomright[(seg_target[:h-1, 1: w] != seg_target[1: h, :w-1]) & (seg_target[: h-1, 1: w] != ignore_index) & (seg_target[1: h, :w-1] != ignore_index)] = 1
        # combine
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (edge_width, edge_width))
        edge = cv2.dilate(edge, kernel)
        # add into sample_meta
        sample_meta['edge_target'] = edge
        # return
        return sample_meta


'''Normalize'''
class Normalize(object):
    def __init__(self, mean, std, to_rgb=True):
        # assert
        assert isinstance(mean, collections.abc.Sequence) and isinstance(std, collections.abc.Sequence)
        # set attributes
        self.to_rgb = to_rgb
        self.std = np.array(std)
        self.mean = np.array(mean)
    '''call'''
    def __call__(self, sample_meta):
        # normalize
        sample_meta['image'] = sample_meta['image'].astype(np.float32)
        mean = np.float64(self.mean.reshape(1, -1))
        stdinv = 1 / np.float64(self.std.reshape(1, -1))
        if self.to_rgb: cv2.cvtColor(sample_meta['image'], cv2.COLOR_BGR2RGB, sample_meta['image'])
        cv2.subtract(sample_meta['image'], mean, sample_meta['image'])
        cv2.multiply(sample_meta['image'], stdinv, sample_meta['image'])
        # return
        return sample_meta


'''Padding'''
class Padding(object):
    def __init__(self, output_size, data_type='numpy', image_fill_value=0, seg_target_fill_value=255, edge_target_fill_value=255, img2aug_pos_mapper_fill_value=-1, output_size_auto_adaptive=True):
        # assert
        assert data_type in ['numpy', 'tensor']
        assert isinstance(output_size, int) or (isinstance(output_size, collections.abc.Sequence) and len(output_size) == 2)
        # set attributes
        self.data_type = data_type
        self.image_fill_value = image_fill_value
        self.seg_target_fill_value = seg_target_fill_value
        self.edge_target_fill_value = edge_target_fill_value
        self.img2aug_pos_mapper_fill_value = img2aug_pos_mapper_fill_value
        self.output_size_auto_adaptive = output_size_auto_adaptive
        self.output_size = (output_size, output_size) if isinstance(output_size, int) else output_size
    '''call'''
    def __call__(self, sample_meta):
        # prepare
        output_size = self.output_size
        if self.output_size_auto_adaptive:
            if self.data_type == 'numpy':
                h_ori, w_ori = sample_meta['image'].shape[:2]
            else:
                h_ori, w_ori = sample_meta['image'].shape[1:]
            h_out, w_out = output_size
            if (h_ori > w_ori and h_out < w_out) or (h_ori < w_ori and h_out > w_out):
                output_size = (w_out, h_out)
        # calculate top, bottom, left, right
        if self.data_type == 'numpy':
            h_ori, w_ori = sample_meta['image'].shape[:2]
        else:
            h_ori, w_ori = sample_meta['image'].shape[1:]
        top = (output_size[0] - h_ori) // 2
        bottom = output_size[0] - h_ori - top
        left = (output_size[1] - w_ori) // 2
        right = output_size[1] - w_ori - left
        # padding
        for key in ['image', 'seg_target', 'edge_target', 'img2aug_pos_mapper']:
            sample_meta = self.padding(key, sample_meta, top, bottom, left, right, getattr(self, f'{key}_fill_value'), self.data_type)
        # return
        return sample_meta
    '''padding'''
    @staticmethod
    def padding(key, sample_meta, top, bottom, left, right, fill_value, data_type):
        if data_type == 'numpy':
            if key in ['image']:
                return Padding.numpypadding(key, sample_meta, top, bottom, left, right, [fill_value, fill_value, fill_value])
            else:
                return Padding.numpypadding(key, sample_meta, top, bottom, left, right, [fill_value])
        else:
            return Padding.tensorpadding(key, sample_meta, top, bottom, left, right, fill_value)
    '''numpypadding'''
    @staticmethod
    def numpypadding(key, sample_meta, top, bottom, left, right, fill_value):
        if key not in sample_meta: return sample_meta
        sample_meta[key] = cv2.copyMakeBorder(sample_meta[key], top, bottom, left, right, cv2.BORDER_CONSTANT, value=fill_value)
        return sample_meta
    '''tensorpadding'''
    @staticmethod
    def tensorpadding(key, sample_meta, top, bottom, left, right, fill_value):
        if key not in sample_meta: return sample_meta
        sample_meta[key] = F.pad(sample_meta[key], pad=(left, right, top, bottom), value=fill_value)
        return sample_meta


'''ToTensor'''
class ToTensor(object):
    '''call'''
    def __call__(self, sample_meta):
        # to tensor
        if 'image' in sample_meta:
            sample_meta['image'] = torch.from_numpy((sample_meta['image'].transpose((2, 0, 1))).astype(np.float32))
        if 'edge_target' in sample_meta:
            sample_meta['edge_target'] = torch.from_numpy(sample_meta['edge_target'].astype(np.float32))
        if 'seg_target' in sample_meta:
            sample_meta['seg_target'] = torch.from_numpy(sample_meta['seg_target'].astype(np.float32))
        if 'img2aug_pos_mapper' in sample_meta:
            sample_meta['img2aug_pos_mapper'] = torch.from_numpy(sample_meta['img2aug_pos_mapper'].astype(np.float32))
        # return
        return sample_meta


'''Compose'''
class Compose(object):
    def __init__(self, transforms, record_img2aug_pos_mapper=False):
        self.transforms = transforms
        self.record_img2aug_pos_mapper = record_img2aug_pos_mapper
    '''call'''
    def __call__(self, sample_meta):
        if 'img2aug_pos_mapper' not in sample_meta:
            sample_meta['img2aug_pos_mapper'] = np.arange(0, sample_meta['height'] * sample_meta['width']).reshape(sample_meta['height'], sample_meta['width'])
        for transform in self.transforms:
            sample_meta = transform(sample_meta)
        return sample_meta


'''DataTransformBuilder'''
class DataTransformBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'Resize': Resize, 'RandomCrop': RandomCrop, 'RandomFlip': RandomFlip, 'RandomRotation': RandomRotation, 'EdgeExtractor': EdgeExtractor,
        'PhotoMetricDistortion': PhotoMetricDistortion, 'Padding': Padding, 'ToTensor': ToTensor, 'ResizeShortestEdge': ResizeShortestEdge,
        'Normalize': Normalize, 'RandomChoiceResize': RandomChoiceResize, 'Rerange': Rerange, 'CLAHE': CLAHE, 'RandomCutOut': RandomCutOut, 
        'AlbumentationsWrapper': AlbumentationsWrapper, 'RGB2Gray': RGB2Gray, 'AdjustGamma': AdjustGamma, 'RandomGaussianBlur': RandomGaussianBlur,
        'RandomShortestEdgeResize': RandomShortestEdgeResize, 'PILRandomGaussianBlur': PILRandomGaussianBlur,
    }
    '''build'''
    def build(self, transform_cfg):
        return super().build(transform_cfg)


'''BuildDataTransform'''
BuildDataTransform = DataTransformBuilder().build