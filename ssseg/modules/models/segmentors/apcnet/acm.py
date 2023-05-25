'''
Function:
    Implementation of AdaptiveContextModule
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...backbones import BuildActivation, BuildNormalization


'''AdaptiveContextModule'''
class AdaptiveContextModule(nn.Module):
    def __init__(self, in_channels, out_channels, pool_scale, align_corners, norm_cfg=None, act_cfg=None):
        super(AdaptiveContextModule, self).__init__()
        self.pool_scale = pool_scale
        self.align_corners = align_corners
        self.pooled_redu_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        self.input_redu_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        self.global_info = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        self.gla = nn.Conv2d(out_channels, pool_scale**2, kernel_size=1, stride=1, padding=0)
        self.residual_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
    '''forward'''
    def forward(self, x):
        batch_size = x.size(0)
        pooled_x = F.adaptive_avg_pool2d(x, self.pool_scale)
        x = self.input_redu_conv(x)
        pooled_x = self.pooled_redu_conv(pooled_x)
        pooled_x = pooled_x.view(batch_size, pooled_x.size(1), -1).permute(0, 2, 1).contiguous()
        affinity_matrix = x + F.interpolate(self.global_info(F.adaptive_avg_pool2d(x, 1)), size=x.shape[2:], align_corners=self.align_corners, mode='bilinear')
        affinity_matrix = self.gla(affinity_matrix).permute(0, 2, 3, 1).reshape(batch_size, -1, self.pool_scale**2)
        affinity_matrix = F.sigmoid(affinity_matrix)
        z_out = torch.matmul(affinity_matrix, pooled_x)
        z_out = z_out.permute(0, 2, 1).contiguous()
        z_out = z_out.view(batch_size, z_out.size(1), x.size(2), x.size(3))
        z_out = self.residual_conv(z_out)
        z_out = F.relu(z_out + x)
        z_out = self.fusion_conv(z_out)
        return z_out