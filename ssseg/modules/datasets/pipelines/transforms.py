'''
Function:
    Define the transforms for data augmentations
Author:
    Zhenchao Jin
'''
import cv2
import torch
import numpy as np
import torch.nn.functional as F


'''Resize'''
class Resize(object):
    def __init__(self, output_size, scale_range=(0.5, 2.0), img_interpolation='bilinear', seg_interpolation='nearest', keep_ratio=True, min_size=None):
        # set attribute
        self.output_size = output_size
        if isinstance(output_size, int): self.output_size = (output_size, output_size)
        self.scale_range = scale_range
        self.img_interpolation = img_interpolation
        self.seg_interpolation = seg_interpolation
        self.keep_ratio = keep_ratio
        self.min_size = min_size
        # interpolation to cv2 interpolation
        self.interpolation_dict = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC,
            'area': cv2.INTER_AREA,
            'lanczos': cv2.INTER_LANCZOS4
        }
    '''call'''
    def __call__(self, sample):
        # parse
        image, segmentation = sample['image'].copy(), sample['segmentation'].copy()
        if self.scale_range is not None:
            rand_scale = np.random.random_sample() * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
            output_size = int(self.output_size[0] * rand_scale), int(self.output_size[1] * rand_scale)
        else:
            output_size = self.output_size[0], self.output_size[1]
        # resize image and segmentation
        if self.keep_ratio:
            scale_factor = min(max(output_size) / max(image.shape[:2]), min(output_size) / min(image.shape[:2]))
            dsize = int(image.shape[1] * scale_factor + 0.5), int(image.shape[0] * scale_factor + 0.5)
            if self.min_size is not None and min(dsize) < self.min_size:
                scale_factor = self.min_size / min(image.shape[:2])
                dsize = int(image.shape[1] * scale_factor + 0.5), int(image.shape[0] * scale_factor + 0.5)
            image = cv2.resize(image, dsize=dsize, interpolation=self.interpolation_dict[self.img_interpolation])
            segmentation = cv2.resize(segmentation, dsize=dsize, interpolation=self.interpolation_dict[self.seg_interpolation])            
        else:
            if image.shape[0] > image.shape[1]:
                dsize = min(output_size), max(output_size)
            else:
                dsize = max(output_size), min(output_size)
            image = cv2.resize(image, dsize=dsize, interpolation=self.interpolation_dict[self.img_interpolation])
            segmentation = cv2.resize(segmentation, dsize=dsize, interpolation=self.interpolation_dict[self.seg_interpolation])
        # update and return sample
        sample['image'], sample['segmentation'] = image, segmentation
        return sample


'''RandomCrop'''
class RandomCrop(object):
    def __init__(self, crop_size, ignore_index=255, one_category_max_ratio=0.75):
        self.crop_size = crop_size
        if isinstance(crop_size, int): self.crop_size = (crop_size, crop_size)
        self.ignore_index = ignore_index
        self.one_category_max_ratio = one_category_max_ratio
    '''call'''
    def __call__(self, sample):
        # avoid the cropped image is filled by only one category
        for _ in range(10):
            # --parse
            image, segmentation = sample['image'].copy(), sample['segmentation'].copy()
            h_ori, w_ori = image.shape[:2]
            h_out, w_out = min(self.crop_size[0], h_ori), min(self.crop_size[1], w_ori)
            # --random crop
            top, left = np.random.randint(0, h_ori - h_out + 1), np.random.randint(0, w_ori - w_out + 1)
            image = image[top: top + h_out, left: left + w_out]
            segmentation = segmentation[top: top + h_out, left: left + w_out]
            # --judge
            labels, counts = np.unique(segmentation, return_counts=True)
            counts = counts[labels != self.ignore_index]
            if len(counts) > 1 and np.max(counts) / np.sum(counts) < self.one_category_max_ratio: break
        # update and return sample
        if len(counts) == 0: return sample
        sample['image'], sample['segmentation'] = image, segmentation
        return sample


'''RandomFlip'''
class RandomFlip(object):
    def __init__(self, flip_prob, fix_ann_pairs=None):
        self.flip_prob = flip_prob
        self.fix_ann_pairs = fix_ann_pairs
    '''call'''
    def __call__(self, sample):
        if np.random.rand() > self.flip_prob: return sample
        image, segmentation = sample['image'].copy(), sample['segmentation'].copy()
        image, segmentation = np.flip(image, axis=1), np.flip(segmentation, axis=1)
        if self.fix_ann_pairs:
            for (pair_a, pair_b) in self.fix_ann_pairs:
                pair_a_pos = np.where(segmentation == pair_a)
                pair_b_pos = np.where(segmentation == pair_b)
                segmentation[pair_a_pos[0], pair_a_pos[1]] = pair_b
                segmentation[pair_b_pos[0], pair_b_pos[1]] = pair_a
        sample['image'], sample['segmentation'] = image, segmentation
        return sample


'''PhotoMetricDistortion'''
class PhotoMetricDistortion(object):
    def __init__(self, brightness_delta=32, contrast_range=(0.5, 1.5), saturation_range=(0.5, 1.5), hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
    '''call'''
    def __call__(self, sample):
        image = sample['image'].copy()
        image = self.brightness(image)
        mode = np.random.randint(2)
        if mode == 1: image = self.contrast(image)
        image = self.saturation(image)
        image = self.hue(image)
        if mode == 0: image = self.contrast(image)
        sample['image'] = image
        return sample
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
    def __init__(self, angle_upper=30, rotation_prob=0.5, img_fill_value=0.0, seg_fill_value=255, img_interpolation='bicubic', seg_interpolation='nearest'):
        # set attributes
        self.angle_upper = angle_upper
        self.rotation_prob = rotation_prob
        self.img_fill_value = img_fill_value
        self.seg_fill_value = seg_fill_value
        self.img_interpolation = img_interpolation
        self.seg_interpolation = seg_interpolation
        # interpolation to cv2 interpolation
        self.interpolation_dict = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC,
            'area': cv2.INTER_AREA,
            'lanczos': cv2.INTER_LANCZOS4
        }
    '''call'''
    def __call__(self, sample):
        if np.random.rand() > self.rotation_prob: return sample
        image, segmentation = sample['image'].copy(), sample['segmentation'].copy()
        h_ori, w_ori = image.shape[:2]
        rand_angle = np.random.randint(-self.angle_upper, self.angle_upper)
        matrix = cv2.getRotationMatrix2D(center=(w_ori / 2, h_ori / 2), angle=rand_angle, scale=1)
        image = cv2.warpAffine(image, matrix, (w_ori, h_ori), flags=self.interpolation_dict[self.img_interpolation], borderValue=self.img_fill_value)
        segmentation = cv2.warpAffine(segmentation, matrix, (w_ori, h_ori), flags=self.interpolation_dict[self.seg_interpolation], borderValue=self.seg_fill_value)
        sample['image'], sample['segmentation'] = image, segmentation
        return sample


'''Padding'''
class Padding(object):
    def __init__(self, output_size, data_type='numpy', img_fill_value=0, seg_fill_value=255, output_size_auto_adaptive=True):
        self.output_size = output_size
        if isinstance(output_size, int): self.output_size = (output_size, output_size)
        assert data_type in ['numpy', 'tensor'], 'unsupport data type %s' % data_type
        self.data_type = data_type
        self.img_fill_value = img_fill_value
        self.seg_fill_value = seg_fill_value
        self.output_size_auto_adaptive = output_size_auto_adaptive
    '''call'''
    def __call__(self, sample):
        output_size = self.output_size[0], self.output_size[1]
        if self.output_size_auto_adaptive:
            if self.data_type == 'numpy':
                h_ori, w_ori = sample['image'].shape[:2]
            else:
                h_ori, w_ori = sample['image'].shape[1:]
            h_out, w_out = output_size
            if (h_ori > w_ori and h_out < w_out) or (h_ori < w_ori and h_out > w_out):
                output_size = (w_out, h_out)
        if self.data_type == 'numpy':
            image, segmentation, edge = sample['image'].copy(), sample['segmentation'].copy(), sample['edge'].copy()
            h_ori, w_ori = image.shape[:2]
            top = (output_size[0] - h_ori) // 2
            bottom = output_size[0] - h_ori - top
            left = (output_size[1] - w_ori) // 2
            right = output_size[1] - w_ori - left
            image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[self.img_fill_value, self.img_fill_value, self.img_fill_value])
            segmentation = cv2.copyMakeBorder(segmentation, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[self.seg_fill_value])
            edge = cv2.copyMakeBorder(edge, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[self.seg_fill_value])
            sample['image'], sample['segmentation'], sample['edge'] = image, segmentation, edge
        else:
            image, segmentation, edge = sample['image'], sample['segmentation'], sample['edge']
            h_ori, w_ori = image.shape[1:]
            top = (output_size[0] - h_ori) // 2
            bottom = output_size[0] - h_ori - top
            left = (output_size[1] - w_ori) // 2
            right = output_size[1] - w_ori - left
            image = F.pad(image, pad=(left, right, top, bottom), value=self.img_fill_value)
            segmentation = F.pad(segmentation, pad=(left, right, top, bottom), value=self.seg_fill_value)
            edge = F.pad(edge, pad=(left, right, top, bottom), value=self.seg_fill_value)
            sample['image'], sample['segmentation'], sample['edge'] = image, segmentation, edge
        return sample


'''ToTensor'''
class ToTensor(object):
    '''call'''
    def __call__(self, sample):
        for key in sample.keys():
            if key == 'image':
                sample[key] = torch.from_numpy((sample[key].transpose((2, 0, 1))).astype(np.float32))
            elif key in ['edge', 'groundtruth', 'segmentation']:
                sample[key] = torch.from_numpy(sample[key].astype(np.float32))
        return sample


'''Normalize'''
class Normalize(object):
    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.to_rgb = to_rgb
    '''call'''
    def __call__(self, sample):
        for key in sample.keys():
            if key == 'image':
                image = sample[key].astype(np.float32)
                mean = np.float64(self.mean.reshape(1, -1))
                stdinv = 1 / np.float64(self.std.reshape(1, -1))
                if self.to_rgb: cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)
                cv2.subtract(image, mean, image)
                cv2.multiply(image, stdinv, image)
                sample[key] = image
        return sample


'''Compose'''
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    '''call'''
    def __call__(self, sample, transform_type):
        if transform_type == 'all':
            for transform in self.transforms:
                sample = transform(sample)
        elif transform_type == 'only_totensor_normalize_pad':
            for transform in self.transforms:
                if isinstance(transform, ToTensor) or isinstance(transform, Normalize) or isinstance(transform, Padding):
                    sample = transform(sample)
        elif transform_type == 'without_totensor_normalize_pad':
            for transform in self.transforms:
                if not (isinstance(transform, ToTensor) or isinstance(transform, Normalize) or isinstance(transform, Padding)):
                    sample = transform(sample)
        else:
            raise ValueError('Unsupport transform_type %s' % transform_type)
        return sample