'''
Function:
    Implementation of SAMV2Transforms
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize, Resize, ToTensor


'''SAMV2Transforms'''
class SAMV2Transforms(nn.Module):
    def __init__(self, resolution, mask_threshold, max_hole_area=0.0, max_sprinkle_area=0.0):
        super(SAMV2Transforms, self).__init__()
        self.resolution = resolution
        self.mask_threshold = mask_threshold
        self.max_hole_area = max_hole_area
        self.max_sprinkle_area = max_sprinkle_area
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.to_tensor = ToTensor()
        self.transforms = torch.jit.script(nn.Sequential(
            Resize((self.resolution, self.resolution)), Normalize(self.mean, self.std),
        ))
    '''call'''
    def __call__(self, x):
        x = self.to_tensor(x)
        return self.transforms(x)
    '''forwardbatch'''
    def forwardbatch(self, img_list):
        img_batch = [self.transforms(self.to_tensor(img)) for img in img_list]
        img_batch = torch.stack(img_batch, dim=0)
        return img_batch
    '''transformcoords'''
    def transformcoords(self, coords, normalize=False, orig_hw=None):
        if normalize:
            assert orig_hw is not None
            h, w = orig_hw
            coords = coords.clone()
            coords[..., 0] = coords[..., 0] / w
            coords[..., 1] = coords[..., 1] / h
        coords = coords * self.resolution
        return coords
    '''transformboxes'''
    def transformboxes(self, boxes, normalize=False, orig_hw=None):
        boxes = self.transformcoords(boxes.reshape(-1, 2, 2), normalize, orig_hw)
        return boxes
    '''postprocessmasks'''
    def postprocessmasks(self, masks, orig_hw):
        from .misc import getconnectedcomponents
        masks = masks.float()
        if self.max_hole_area > 0:
            mask_flat = masks.flatten(0, 1).unsqueeze(1)
            labels, areas = getconnectedcomponents(mask_flat <= self.mask_threshold)
            is_hole = (labels > 0) & (areas <= self.max_hole_area)
            is_hole = is_hole.reshape_as(masks)
            masks = torch.where(is_hole, self.mask_threshold + 10.0, masks)
        if self.max_sprinkle_area > 0:
            labels, areas = getconnectedcomponents(mask_flat > self.mask_threshold)
            is_hole = (labels > 0) & (areas <= self.max_sprinkle_area)
            is_hole = is_hole.reshape_as(masks)
            masks = torch.where(is_hole, self.mask_threshold - 10.0, masks)
        masks = F.interpolate(masks, orig_hw, mode='bilinear', align_corners=False)
        return masks