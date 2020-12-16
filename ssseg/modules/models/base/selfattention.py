'''
Function:
    define the self attention block
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


'''self attention block'''
class SelfAttentionBlock(nn.Module):
    def __init__(self, **kwargs):
        super(SelfAttentionBlock, self).__init__()
        # whether use norm
        self.matmul_norm_cfg = kwargs.get('matmul_norm_cfg', {'is_on': True, 'key_channels': 256})
        # downsample layers
        self.query_downsample = kwargs.get('query_downsample', None)
        self.key_downsample = kwargs.get('key_downsample', None)
        # query and key project
        self.query_project = kwargs.get('query_project', None)
        assert self.query_project is not None
        self.key_project = kwargs.get('key_project', None)
        assert self.key_project is not None
        self.value_project = kwargs.get('value_project', None)
        assert self.value_project is not None
        # out project
        self.out_project = kwargs.get('out_project', None)
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
        if self.matmul_norm_cfg['is_on']:
            sim_map = (self.matmul_norm_cfg['key_channels'] ** -0.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.reshape(batch_size, -1, *query_feats.shape[2:])
        if self.out_project is not None:
            context = self.out_project(context)
        return context