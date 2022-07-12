'''
Function:
    Implementation of Encoding Layer: a learnable residual encoder
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


'''Encoding'''
class Encoding(nn.Module):
    def __init__(self, channels, num_codes):
        super(Encoding, self).__init__()
        # init codewords and smoothing factor
        self.channels, self.num_codes = channels, num_codes
        std = 1. / ((num_codes * channels) ** 0.5)
        # [num_codes, channels]
        self.codewords = nn.Parameter(torch.empty(num_codes, channels, dtype=torch.float).uniform_(-std, std), requires_grad=True)
        # [num_codes]
        self.scale = nn.Parameter(torch.empty(num_codes, dtype=torch.float).uniform_(-1, 0), requires_grad=True)
    '''scaled l2'''
    @staticmethod
    def scaledl2(x, codewords, scale):
        batch_size = x.size(0)
        num_codes, channels = codewords.size()
        reshaped_scale = scale.view((1, 1, num_codes))
        expanded_x = x.unsqueeze(2).expand((batch_size, x.size(1), num_codes, channels))
        reshaped_codewords = codewords.view((1, 1, num_codes, channels))
        scaledl2_norm = reshaped_scale * (expanded_x - reshaped_codewords).pow(2).sum(dim=3)
        return scaledl2_norm
    '''aggregate'''
    @staticmethod
    def aggregate(assigment_weights, x, codewords):
        batch_size = x.size(0)
        num_codes, channels = codewords.size()
        reshaped_codewords = codewords.view((1, 1, num_codes, channels))
        expanded_x = x.unsqueeze(2).expand((batch_size, x.size(1), num_codes, channels))
        encoded_feat = (assigment_weights.unsqueeze(3) * (expanded_x - reshaped_codewords)).sum(dim=1)
        return encoded_feat
    '''forward'''
    def forward(self, x):
        assert x.dim() == 4 and x.size(1) == self.channels
        # [batch_size, channels, height, width]
        batch_size = x.size(0)
        # [batch_size, height x width, channels]
        x = x.view(batch_size, self.channels, -1).transpose(1, 2).contiguous()
        # assignment_weights: [batch_size, channels, num_codes]
        assigment_weights = F.softmax(self.scaledl2(x, self.codewords, self.scale), dim=2)
        # aggregate
        encoded_feat = self.aggregate(assigment_weights, x, self.codewords)
        return encoded_feat