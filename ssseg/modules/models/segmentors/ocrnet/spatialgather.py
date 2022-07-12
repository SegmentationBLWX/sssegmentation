'''
Function:
    Implementation of SpatialGatherModule
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


'''SpatialGatherModule'''
class SpatialGatherModule(nn.Module):
    def __init__(self, scale=1):
        super(SpatialGatherModule, self).__init__()
        self.scale = scale
    '''forward'''
    def forward(self, features, probs):
        batch_size, num_classes, h, w = probs.size()
        probs = probs.view(batch_size, num_classes, -1)
        features = features.view(batch_size, features.size(1), -1)
        features = features.permute(0, 2, 1)
        probs = F.softmax(self.scale * probs, dim=2)
        ocr_context = torch.matmul(probs, features)
        ocr_context = ocr_context.permute(0, 2, 1).contiguous().unsqueeze(3)
        return ocr_context