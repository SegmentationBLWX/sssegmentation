'''
Function:
    Implementation of Position Attention Module (PAM)
Author:
    Zhenchao Jin
'''
from ...backbones import Scale
from ..base import SelfAttentionBlock


'''PositionAttentionModule'''
class PositionAttentionModule(SelfAttentionBlock):
    def __init__(self, in_channels, transform_channels):
        super(PositionAttentionModule, self).__init__(
            key_in_channels=in_channels, query_in_channels=in_channels, transform_channels=transform_channels, out_channels=in_channels,
            share_key_query=False, query_downsample=None, key_downsample=None, key_query_num_convs=1, value_out_num_convs=1, key_query_norm=False,
            value_out_norm=False, matmul_norm=False, with_out_project=False, norm_cfg=None, act_cfg=None,
        )
        self.gamma = Scale(scale=0)
    '''forward'''
    def forward(self, x):
        out = super(PositionAttentionModule, self).forward(x, x)
        out = self.gamma(out) + x
        return out