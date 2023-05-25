'''
Function:
    Implementation of Transforms
Author:
    Zhenchao Jin
'''
import cv2
import torch
import collections
import numpy as np
import torch.nn.functional as F


'''_INTERPOLATION_CV2_CONVERTOR'''
_INTERPOLATION_CV2_CONVERTOR = {
    'area': cv2.INTER_AREA, 'bicubic': cv2.INTER_CUBIC, 'bilinear': cv2.INTER_LINEAR, 
    'nearest': cv2.INTER_NEAREST, 'lanczos': cv2.INTER_LANCZOS4,
}


'''Resize'''
class Resize(object):
    def __init__(self, output_size, keep_ratio=True, min_size=None, scale_range=(0.5, 2.0), image_interpolation='bilinear', seg_target_interpolation='nearest'):
        # assert
        assert isinstance(scale_range, collections.Sequence) or scale_range is None
        assert isinstance(output_size, int) or (isinstance(output_size, collections.Sequence) and len(output_size) == 2)
        # set attributes
        self.min_size = min_size
        self.keep_ratio = keep_ratio
        self.scale_range = scale_range
        self.image_interpolation = _INTERPOLATION_CV2_CONVERTOR[image_interpolation]
        self.seg_target_interpolation = _INTERPOLATION_CV2_CONVERTOR[seg_target_interpolation]
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
        assert isinstance(crop_size, int) or (isinstance(crop_size, collections.Sequence) and len(crop_size) == 2)
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
            # --judge
            labels, counts = np.unique(seg_target, return_counts=True)
            counts = counts[labels != self.ignore_index]
            if len(counts) > 1 and np.max(counts) / np.sum(counts) < self.one_category_max_ratio:
                break
        # update
        if len(counts) == 0: return sample_meta
        sample_meta['image'], sample_meta['seg_target'] = image, seg_target
        # return
        return sample_meta
    '''crop'''
    @staticmethod
    def crop(image, top, left, h_out, w_out):
        image = image[top: top + h_out, left: left + w_out]
        return image


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
        if np.random.rand() > self.flip_prob: return sample_meta
        sample_meta = self.flip('image', sample_meta)
        sample_meta = self.flip('seg_target', sample_meta)
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
        assert isinstance(contrast_range, collections.Sequence) and len(contrast_range) == 2
        assert isinstance(saturation_range, collections.Sequence) and len(saturation_range) == 2
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
    def __init__(self, rotation_prob=0.5, angle_upper=30, image_fill_value=0.0, seg_target_fill_value=255, image_interpolation='bicubic', seg_target_interpolation='nearest'):
        # assert
        assert isinstance(rotation_prob, float)
        # set attributes
        self.rotation_prob = rotation_prob
        self.angle_upper = angle_upper
        self.image_fill_value = image_fill_value
        self.seg_target_fill_value = seg_target_fill_value
        self.image_interpolation = _INTERPOLATION_CV2_CONVERTOR[image_interpolation]
        self.seg_target_interpolation = _INTERPOLATION_CV2_CONVERTOR[seg_target_interpolation]
    '''call'''
    def __call__(self, sample_meta):
        # prepare
        if np.random.rand() > self.rotation_prob: return sample_meta
        h_ori, w_ori = sample_meta['image'].shape[:2]
        rand_angle = np.random.randint(-self.angle_upper, self.angle_upper)
        matrix = cv2.getRotationMatrix2D(center=(w_ori / 2, h_ori / 2), angle=rand_angle, scale=1)
        # rotate
        sample_meta = self.rotate('image', sample_meta, matrix, w_ori, h_ori, self.image_interpolation, self.image_fill_value)
        sample_meta = self.rotate('seg_target', sample_meta, matrix, w_ori, h_ori, self.seg_target_interpolation, self.seg_target_fill_value)
        # return
        return sample_meta
    '''rotate'''
    @staticmethod
    def rotate(key, sample_meta, matrix, w_ori, h_ori, interpolation, fill_value):
        if key not in sample_meta: return sample_meta
        sample_meta[key] = cv2.warpAffine(sample_meta[key], matrix, (w_ori, h_ori), flags=interpolation, borderValue=fill_value)
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
        assert isinstance(mean, collections.Sequence) and isinstance(std, collections.Sequence)
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
    def __init__(self, output_size, data_type='numpy', image_fill_value=0, seg_target_fill_value=255, edge_target_fill_value=255, output_size_auto_adaptive=True):
        # assert
        assert data_type in ['numpy', 'tensor']
        assert isinstance(output_size, int) or (isinstance(output_size, collections.Sequence) and len(output_size) == 2)
        # set attributes
        self.data_type = data_type
        self.image_fill_value = image_fill_value
        self.seg_target_fill_value = seg_target_fill_value
        self.edge_target_fill_value = edge_target_fill_value
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
        for key in ['image', 'seg_target', 'edge_target']:
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
        # return
        return sample_meta


'''Compose'''
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    '''call'''
    def __call__(self, sample_meta):
        for transform in self.transforms:
            sample_meta = transform(sample_meta)
        return sample_meta