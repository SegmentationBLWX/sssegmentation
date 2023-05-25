'''
Function:
    Implementation of FeaturesMemory
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from ..base import SelfAttentionBlock
from ...backbones import BuildActivation, BuildNormalization


'''FeaturesMemory'''
class FeaturesMemory(nn.Module):
    def __init__(self, num_classes, feats_channels, transform_channels, out_channels, use_context_within_image=True, 
                 num_feats_per_cls=1, use_hard_aggregate=False, norm_cfg=None, act_cfg=None):
        super(FeaturesMemory, self).__init__()
        assert num_feats_per_cls > 0, 'num_feats_per_cls should be larger than 0'
        # set attributes
        self.num_classes = num_classes
        self.feats_channels = feats_channels
        self.transform_channels = transform_channels
        self.out_channels = out_channels
        self.num_feats_per_cls = num_feats_per_cls
        self.use_context_within_image = use_context_within_image
        self.use_hard_aggregate = use_hard_aggregate
        # init memory
        self.memory = nn.Parameter(torch.zeros(num_classes, num_feats_per_cls, feats_channels, dtype=torch.float), requires_grad=False)
        # define self_attention module
        if self.num_feats_per_cls > 1:
            self.self_attentions = nn.ModuleList()
            for _ in range(self.num_feats_per_cls):
                self_attention = SelfAttentionBlock(
                    key_in_channels=feats_channels,
                    query_in_channels=feats_channels,
                    transform_channels=transform_channels,
                    out_channels=feats_channels,
                    share_key_query=False,
                    query_downsample=None,
                    key_downsample=None,
                    key_query_num_convs=2,
                    value_out_num_convs=1,
                    key_query_norm=True,
                    value_out_norm=True,
                    matmul_norm=True,
                    with_out_project=True,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
                self.self_attentions.append(self_attention)
            self.fuse_memory_conv = nn.Sequential(
                nn.Conv2d(feats_channels * self.num_feats_per_cls, feats_channels, kernel_size=1, stride=1, padding=0, bias=False),
                BuildNormalization(placeholder=feats_channels, norm_cfg=norm_cfg),
                BuildActivation(act_cfg),
            )
        else:
            self.self_attention = SelfAttentionBlock(
                key_in_channels=feats_channels,
                query_in_channels=feats_channels,
                transform_channels=transform_channels,
                out_channels=feats_channels,
                share_key_query=False,
                query_downsample=None,
                key_downsample=None,
                key_query_num_convs=2,
                value_out_num_convs=1,
                key_query_norm=True,
                value_out_norm=True,
                matmul_norm=True,
                with_out_project=True,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
        # whether need to fuse the contextual information within the input image
        self.bottleneck = nn.Sequential(
            nn.Conv2d(feats_channels * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        if use_context_within_image:
            self.self_attention_ms = SelfAttentionBlock(
                key_in_channels=feats_channels,
                query_in_channels=feats_channels,
                transform_channels=transform_channels,
                out_channels=feats_channels,
                share_key_query=False,
                query_downsample=None,
                key_downsample=None,
                key_query_num_convs=2,
                value_out_num_convs=1,
                key_query_norm=True,
                value_out_norm=True,
                matmul_norm=True,
                with_out_project=True,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
            self.bottleneck_ms = nn.Sequential(
                nn.Conv2d(feats_channels * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
                BuildActivation(act_cfg),
            )
    '''forward'''
    def forward(self, feats, preds=None, feats_ms=None):
        batch_size, num_channels, h, w = feats.size()
        # extract the history features
        # --(B, num_classes, H, W) --> (B*H*W, num_classes)
        weight_cls = preds.permute(0, 2, 3, 1).contiguous()
        weight_cls = weight_cls.reshape(-1, self.num_classes)
        weight_cls = F.softmax(weight_cls, dim=-1)
        if self.use_hard_aggregate:
            labels = weight_cls.argmax(-1).reshape(-1, 1)
            onehot = torch.zeros_like(weight_cls).scatter_(1, labels.long(), 1)
            weight_cls = onehot
        # --(B*H*W, num_classes) * (num_classes, C) --> (B*H*W, C)
        selected_memory_list = []
        for idx in range(self.num_feats_per_cls):
            memory = self.memory.data[:, idx, :]
            selected_memory = torch.matmul(weight_cls, memory)
            selected_memory_list.append(selected_memory.unsqueeze(1))
        # calculate selected_memory according to the num_feats_per_cls
        if self.num_feats_per_cls > 1:
            relation_selected_memory_list = []
            for idx, selected_memory in enumerate(selected_memory_list):
                # --(B*H*W, C) --> (B, H, W, C)
                selected_memory = selected_memory.view(batch_size, h, w, num_channels)
                # --(B, H, W, C) --> (B, C, H, W)
                selected_memory = selected_memory.permute(0, 3, 1, 2).contiguous()
                # --append
                relation_selected_memory_list.append(self.self_attentions[idx](feats, selected_memory))
            # --concat
            selected_memory = torch.cat(relation_selected_memory_list, dim=1)
            selected_memory = self.fuse_memory_conv(selected_memory)
        else:
            assert len(selected_memory_list) == 1
            selected_memory = selected_memory_list[0].squeeze(1)
            # --(B*H*W, C) --> (B, H, W, C)
            selected_memory = selected_memory.view(batch_size, h, w, num_channels)
            # --(B, H, W, C) --> (B, C, H, W)
            selected_memory = selected_memory.permute(0, 3, 1, 2).contiguous()
            # --feed into the self attention module
            selected_memory = self.self_attention(feats, selected_memory)
        # return
        memory_output = self.bottleneck(torch.cat([feats, selected_memory], dim=1))
        if self.use_context_within_image:
            feats_ms = self.self_attention_ms(feats, feats_ms)
            memory_output = self.bottleneck_ms(torch.cat([feats_ms, memory_output], dim=1))
        return self.memory.data, memory_output
    '''update'''
    def update(self, features, segmentation, ignore_index=255, strategy='cosine_similarity', momentum_cfg=None, learning_rate=None):
        assert strategy in ['mean', 'cosine_similarity']
        batch_size, num_channels, h, w = features.size()
        momentum = momentum_cfg['base_momentum']
        if momentum_cfg['adjust_by_learning_rate']:
            momentum = momentum_cfg['base_momentum'] / momentum_cfg['base_lr'] * learning_rate
        # use features to update memory
        segmentation = segmentation.long()
        features = features.permute(0, 2, 3, 1).contiguous()
        features = features.view(batch_size * h * w, num_channels)
        clsids = segmentation.unique()
        for clsid in clsids:
            if clsid == ignore_index: continue
            # --(B, H, W) --> (B*H*W,)
            seg_cls = segmentation.view(-1)
            # --extract the corresponding feats: (K, C)
            feats_cls = features[seg_cls == clsid]
            # --init memory by using extracted features
            need_update = True
            for idx in range(self.num_feats_per_cls):
                if (self.memory[clsid][idx] == 0).sum() == self.feats_channels:
                    self.memory[clsid][idx].data.copy_(feats_cls.mean(0))
                    need_update = False
                    break
            if not need_update: continue
            # --update according to the selected strategy
            if self.num_feats_per_cls == 1:
                if strategy == 'mean':
                    feats_cls = feats_cls.mean(0)
                elif strategy == 'cosine_similarity':
                    similarity = F.cosine_similarity(feats_cls, self.memory[clsid].data.expand_as(feats_cls))
                    weight = (1 - similarity) / (1 - similarity).sum()
                    feats_cls = (feats_cls * weight.unsqueeze(-1)).sum(0)
                feats_cls = (1 - momentum) * self.memory[clsid].data + momentum * feats_cls.unsqueeze(0)
                self.memory[clsid].data.copy_(feats_cls)
            else:
                assert strategy in ['cosine_similarity']
                # ----(K, C) * (C, num_feats_per_cls) --> (K, num_feats_per_cls)
                relation = torch.matmul(
                    F.normalize(feats_cls, p=2, dim=1), 
                    F.normalize(self.memory[clsid].data.permute(1, 0).contiguous(), p=2, dim=0),
                )
                argmax = relation.argmax(dim=1)
                # ----for saving memory during training
                for idx in range(self.num_feats_per_cls):
                    mask = (argmax == idx)
                    feats_cls_iter = feats_cls[mask]
                    memory_cls_iter = self.memory[clsid].data[idx].unsqueeze(0).expand_as(feats_cls_iter)
                    similarity = F.cosine_similarity(feats_cls_iter, memory_cls_iter)
                    weight = (1 - similarity) / (1 - similarity).sum()
                    feats_cls_iter = (feats_cls_iter * weight.unsqueeze(-1)).sum(0)
                    self.memory[clsid].data[idx].copy_(self.memory[clsid].data[idx] * (1 - momentum) + feats_cls_iter * momentum)
        # syn the memory
        if dist.is_available() and dist.is_initialized():
            memory = self.memory.data.clone()
            dist.all_reduce(memory.div_(dist.get_world_size()))
            self.memory = nn.Parameter(memory, requires_grad=False)