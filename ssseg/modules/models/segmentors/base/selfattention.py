'''
Function:
    Implementation of SelfAttentionBlock
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...backbones import BuildActivation, BuildNormalization


'''SelfAttentionBlock'''
class SelfAttentionBlock(nn.Module):
    def __init__(self, key_in_channels, query_in_channels, transform_channels, out_channels, share_key_query, 
                 query_downsample, key_downsample, key_query_num_convs, value_out_num_convs, key_query_norm, 
                 value_out_norm, matmul_norm, with_out_project, norm_cfg=None, act_cfg=None):
        super(SelfAttentionBlock, self).__init__()
        # key project
        self.key_project = self.buildproject(
            in_channels=key_in_channels, out_channels=transform_channels, num_convs=key_query_num_convs,
            use_norm=key_query_norm, norm_cfg=norm_cfg, act_cfg=act_cfg,
        )
        # query project
        if share_key_query:
            assert key_in_channels == query_in_channels
            self.query_project = self.key_project
        else:
            self.query_project = self.buildproject(
                in_channels=query_in_channels, out_channels=transform_channels, num_convs=key_query_num_convs,
                use_norm=key_query_norm, norm_cfg=norm_cfg, act_cfg=act_cfg,
            )
        # value project
        self.value_project = self.buildproject(
            in_channels=key_in_channels, out_channels=transform_channels if with_out_project else out_channels, num_convs=value_out_num_convs,
            use_norm=value_out_norm, norm_cfg=norm_cfg, act_cfg=act_cfg,
        )
        # out project
        self.out_project = None
        if with_out_project:
            self.out_project = self.buildproject(
                in_channels=transform_channels, out_channels=out_channels, num_convs=value_out_num_convs,
                use_norm=value_out_norm, norm_cfg=norm_cfg, act_cfg=act_cfg,
            )
        # downsample
        self.query_downsample = query_downsample
        self.key_downsample = key_downsample
        self.matmul_norm = matmul_norm
        self.transform_channels = transform_channels
    '''forward'''
    def forward(self, query_feats, key_feats):
        batch_size = query_feats.size(0)
        query = self.query_project(query_feats)
        if self.query_downsample is not None: query = self.query_downsample(query)
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous()
        key = self.key_project(key_feats)
        value = self.value_project(key_feats)
        if self.key_downsample is not None:
            key = self.key_downsample(key)
            value = self.key_downsample(value)
        key = key.reshape(*key.shape[:2], -1)
        value = value.reshape(*value.shape[:2], -1)
        value = value.permute(0, 2, 1).contiguous()
        sim_map = torch.matmul(query, key)
        if self.matmul_norm:
            sim_map = (self.transform_channels ** -0.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.reshape(batch_size, -1, *query_feats.shape[2:])
        if self.out_project is not None:
            context = self.out_project(context)
        return context
    '''build project'''
    def buildproject(self, in_channels, out_channels, num_convs, use_norm, norm_cfg, act_cfg):
        if use_norm:
            convs = [nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
                BuildActivation(act_cfg),
            )]
            for _ in range(num_convs - 1):
                convs.append(nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                    BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
                    BuildActivation(act_cfg),
                ))
        else:
            convs = [nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)]
            for _ in range(num_convs - 1):
                convs.append(
                    nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
                )
        if len(convs) > 1: return nn.Sequential(*convs)
        return convs[0]