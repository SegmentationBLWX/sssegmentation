'''
Function:
    Implementation of Feature2Pyramid
Author:
    Zhenchao Jin
'''
import torch.nn as nn
from ...backbones import BuildNormalization


'''Feature2Pyramid'''
class Feature2Pyramid(nn.Module):
    def __init__(self, embed_dim, rescales=[4, 2, 1, 0.5], norm_cfg=None):
        super(Feature2Pyramid, self).__init__()
        self.rescales = rescales
        self.upsample_4x = None
        for k in self.rescales:
            if k == 4:
                self.upsample_4x = nn.Sequential(
                    nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                    BuildNormalization(placeholder=embed_dim, norm_cfg=norm_cfg),
                    nn.GELU(),
                    nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                )
            elif k == 2:
                self.upsample_2x = nn.Sequential(
                    nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2)
                )
            elif k == 1:
                self.identity = nn.Identity()
            elif k == 0.5:
                self.downsample_2x = nn.MaxPool2d(kernel_size=2, stride=2)
            elif k == 0.25:
                self.downsample_4x = nn.MaxPool2d(kernel_size=4, stride=4)
            else:
                raise KeyError(f'invalid {k} for feature2pyramid')
    '''forward'''
    def forward(self, inputs):
        assert len(inputs) == len(self.rescales)
        outputs = []
        if self.upsample_4x is not None:
            ops = [self.upsample_4x, self.upsample_2x, self.identity, self.downsample_2x]
        else:
            ops = [self.upsample_2x, self.identity, self.downsample_2x, self.downsample_4x]
        for i in range(len(inputs)):
            outputs.append(ops[i](inputs[i]))
        return tuple(outputs)