'''
Function:
    Implementation of Context Encoding Module
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoding import Encoding
from ...backbones import BuildActivation, BuildNormalization


'''ContextEncoding'''
class ContextEncoding(nn.Module):
    def __init__(self, in_channels, num_codes, norm_cfg=None, act_cfg=None):
        super(ContextEncoding, self).__init__()
        self.encoding_project = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(placeholder=in_channels, norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        encoding_norm_cfg = copy.deepcopy(norm_cfg)
        encoding_norm_cfg['type'] = encoding_norm_cfg['type'].replace('2d', '1d')
        self.encoding = nn.Sequential(
            Encoding(channels=in_channels, num_codes=num_codes),
            BuildNormalization(placeholder=num_codes, norm_cfg=encoding_norm_cfg),
            BuildActivation(act_cfg),
        )
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid()
        )
    '''forward'''
    def forward(self, x):
        encoding_projection = self.encoding_project(x)
        encoding_feat = self.encoding(encoding_projection).mean(dim=1)
        batch_size, channels, _, _ = x.size()
        gamma = self.fc(encoding_feat)
        y = gamma.view(batch_size, channels, 1, 1)
        output = F.relu_(x + x * y)
        return encoding_feat, output