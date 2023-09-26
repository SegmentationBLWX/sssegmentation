'''
Function:
    Implementation of Transforms
Author:
    Zhenchao Jin
'''
import torch
import numpy as np
from copy import deepcopy
from torch.nn import functional as F
from torchvision.transforms.functional import resize, to_pil_image


'''ResizeLongestSide'''
class ResizeLongestSide:
    def __init__(self, target_length):
        self.target_length = target_length
    '''applyimage'''
    def applyimage(self, image):
        target_size = self.getpreprocessshape(image.shape[0], image.shape[1], self.target_length)
        return np.array(resize(to_pil_image(image), target_size))
    '''applycoords'''
    def applycoords(self, coords, original_size):
        old_h, old_w = original_size
        new_h, new_w = self.getpreprocessshape(original_size[0], original_size[1], self.target_length)
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords
    '''applyboxes'''
    def applyboxes(self, boxes, original_size):
        boxes = self.applycoords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)
    '''applyimagetorch'''
    def applyimagetorch(self, image):
        # expects an image in BCHW format. may not exactly match apply image.
        target_size = self.getpreprocessshape(image.shape[2], image.shape[3], self.target_length)
        return F.interpolate(image, target_size, mode='bilinear', align_corners=False, antialias=True)
    '''applycoordstorch'''
    def applycoordstorch(self, coords, original_size):
        old_h, old_w = original_size
        new_h, new_w = self.getpreprocessshape(original_size[0], original_size[1], self.target_length)
        coords = deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords
    '''applyboxestorch'''
    def applyboxestorch(self, boxes, original_size):
        boxes = self.applycoordstorch(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)
    '''getpreprocessshape'''
    @staticmethod
    def getpreprocessshape(oldh, oldw, long_side_length):
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)
