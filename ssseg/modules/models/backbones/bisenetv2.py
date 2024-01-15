'''
Function:
    Implementation of BiSeNetV2
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...utils import loadpretrainedweights
from .bricks import BuildNormalization, BuildActivation, DepthwiseSeparableConv2d


'''DEFAULT_MODEL_URLS'''
DEFAULT_MODEL_URLS = {}
'''AUTO_ASSERT_STRUCTURE_TYPES'''
AUTO_ASSERT_STRUCTURE_TYPES = {}


'''DetailBranch'''
class DetailBranch(nn.Module):
    def __init__(self, detail_channels=(64, 64, 128), in_channels=3, norm_cfg=None, act_cfg=None):
        super(DetailBranch, self).__init__()
        detail_branch = []
        for i in range(len(detail_channels)):
            if i == 0:
                detail_branch.append(nn.Sequential(
                    nn.Conv2d(in_channels, detail_channels[i], kernel_size=3, stride=2, padding=1, bias=False),
                    BuildNormalization(placeholder=detail_channels[i], norm_cfg=norm_cfg),
                    BuildActivation(act_cfg),
                    nn.Conv2d(detail_channels[i], detail_channels[i], kernel_size=3, stride=1, padding=1, bias=False),
                    BuildNormalization(placeholder=detail_channels[i], norm_cfg=norm_cfg),
                    BuildActivation(act_cfg),
                ))
            else:
                detail_branch.append(nn.Sequential(
                    nn.Conv2d(detail_channels[i - 1], detail_channels[i], kernel_size=3, stride=2, padding=1, bias=False),
                    BuildNormalization(placeholder=detail_channels[i], norm_cfg=norm_cfg),
                    BuildActivation(act_cfg),
                    nn.Conv2d(detail_channels[i], detail_channels[i], kernel_size=3, stride=1, padding=1, bias=False),
                    BuildNormalization(placeholder=detail_channels[i], norm_cfg=norm_cfg),
                    BuildActivation(act_cfg),
                    nn.Conv2d(detail_channels[i], detail_channels[i], kernel_size=3, stride=1, padding=1, bias=False),
                    BuildNormalization(placeholder=detail_channels[i], norm_cfg=norm_cfg),
                    BuildActivation(act_cfg),
                ))
        self.detail_branch = nn.ModuleList(detail_branch)
    '''forward'''
    def forward(self, x):
        for stage in self.detail_branch:
            x = stage(x)
        return x


'''StemBlock'''
class StemBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=16, norm_cfg=None, act_cfg=None):
        super(StemBlock, self).__init__()
        self.conv_first = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        self.convs = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 2, kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(placeholder=out_channels//2, norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.fuse_last = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
    '''forward'''
    def forward(self, x):
        x = self.conv_first(x)
        x_left = self.convs(x)
        x_right = self.pool(x)
        x = self.fuse_last(torch.cat([x_left, x_right], dim=1))
        return x


'''GELayer'''
class GELayer(nn.Module):
    def __init__(self, in_channels, out_channels, exp_ratio=6, stride=1, norm_cfg=None, act_cfg=None):
        super(GELayer, self).__init__()
        mid_channel = in_channels * exp_ratio
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(placeholder=in_channels, norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        if stride == 1:
            self.dwconv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channel, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
                BuildNormalization(placeholder=mid_channel, norm_cfg=norm_cfg),
                BuildActivation(act_cfg),
            )
            self.shortcut = None
        else:
            self.dwconv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channel, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
                BuildNormalization(placeholder=mid_channel, norm_cfg=norm_cfg),
                nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1, groups=mid_channel, bias=False),
                BuildNormalization(placeholder=mid_channel, norm_cfg=norm_cfg),
                BuildActivation(act_cfg),
            )
            self.shortcut = nn.Sequential(DepthwiseSeparableConv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1,
                dw_norm_cfg=norm_cfg, dw_act_cfg=None, pw_norm_cfg=norm_cfg, pw_act_cfg=None,
            ))
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
        )
        self.act = BuildActivation(act_cfg)
    '''forward'''
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.conv2(x)
        if self.shortcut is not None:
            shortcut = self.shortcut(identity)
            x = x + shortcut
        else:
            x = x + identity
        x = self.act(x)
        return x


'''CEBlock'''
class CEBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=16, norm_cfg=None, act_cfg=None):
        super(CEBlock, self).__init__()
        # set attrs
        self.in_channels = in_channels
        self.out_channels = out_channels
        # define modules
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            BuildNormalization(placeholder=in_channels, norm_cfg=norm_cfg),
        )
        self.conv_gap = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        self.conv_last = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
    '''forward'''
    def forward(self, x):
        identity = x
        x = self.gap(x)
        x = self.conv_gap(x)
        x = identity + x
        x = self.conv_last(x)
        return x


'''SemanticBranch'''
class SemanticBranch(nn.Module):
    def __init__(self, semantic_channels=(16, 32, 64, 128), in_channels=3, exp_ratio=6, norm_cfg=None, act_cfg=None):
        super(SemanticBranch, self).__init__()
        # set attrs
        self.in_channels = in_channels
        self.semantic_channels = semantic_channels
        self.semantic_stages = []
        # define modules
        for i in range(len(semantic_channels)):
            stage_name = f'stage{i + 1}'
            self.semantic_stages.append(stage_name)
            if i == 0:
                self.add_module(stage_name, StemBlock(in_channels, semantic_channels[i], norm_cfg=norm_cfg, act_cfg=act_cfg))
            elif i == (len(semantic_channels) - 1):
                self.add_module(stage_name, nn.Sequential(
                    GELayer(semantic_channels[i - 1], semantic_channels[i], exp_ratio, 2, norm_cfg=norm_cfg, act_cfg=act_cfg),
                    GELayer(semantic_channels[i], semantic_channels[i], exp_ratio, 1, norm_cfg=norm_cfg, act_cfg=act_cfg),
                    GELayer(semantic_channels[i], semantic_channels[i], exp_ratio, 1, norm_cfg=norm_cfg, act_cfg=act_cfg),
                    GELayer(semantic_channels[i], semantic_channels[i], exp_ratio, 1, norm_cfg=norm_cfg, act_cfg=act_cfg),
                ))
            else:
                self.add_module(stage_name, nn.Sequential(
                    GELayer(semantic_channels[i - 1], semantic_channels[i], exp_ratio, 2, norm_cfg=norm_cfg, act_cfg=act_cfg),
                    GELayer(semantic_channels[i], semantic_channels[i], exp_ratio, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
                ))
        self.add_module(f'stage{len(semantic_channels)}_CEBlock',
            CEBlock(semantic_channels[-1], semantic_channels[-1], norm_cfg=norm_cfg, act_cfg=act_cfg),
        )
        self.semantic_stages.append(f'stage{len(semantic_channels)}_CEBlock')
    '''forward'''
    def forward(self, x):
        semantic_outs = []
        for stage_name in self.semantic_stages:
            semantic_stage = getattr(self, stage_name)
            x = semantic_stage(x)
            semantic_outs.append(x)
        return semantic_outs


'''BGALayer'''
class BGALayer(nn.Module):
    def __init__(self, out_channels=128, align_corners=False, norm_cfg=None, act_cfg=None):
        super(BGALayer, self).__init__()
        # set attrs
        self.out_channels = out_channels
        self.align_corners = align_corners
        # define modules
        self.detail_dwconv = nn.Sequential(DepthwiseSeparableConv2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
            dw_norm_cfg=norm_cfg, dw_act_cfg=None, pw_norm_cfg=None, pw_act_cfg=None,
        ))
        self.detail_down = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False),
        )
        self.semantic_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
        )
        self.semantic_dwconv = nn.Sequential(DepthwiseSeparableConv2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
            dw_norm_cfg=norm_cfg, dw_act_cfg=None, pw_norm_cfg=None, pw_act_cfg=None,
        ))
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
    '''forward'''
    def forward(self, x_d, x_s):
        detail_dwconv = self.detail_dwconv(x_d)
        detail_down = self.detail_down(x_d)
        semantic_conv = self.semantic_conv(x_s)
        semantic_dwconv = self.semantic_dwconv(x_s)
        semantic_conv = F.interpolate(semantic_conv, size=detail_dwconv.shape[2:], mode='bilinear', align_corners=self.align_corners)
        fuse_1 = detail_dwconv * torch.sigmoid(semantic_conv)
        fuse_2 = detail_down * torch.sigmoid(semantic_dwconv)
        fuse_2 = F.interpolate(fuse_2, size=fuse_1.shape[2:], mode='bilinear', align_corners=self.align_corners)
        output = self.conv(fuse_1 + fuse_2)
        return output


'''BiSeNetV2'''
class BiSeNetV2(nn.Module):
    def __init__(self, structure_type, in_channels=3, detail_channels=(64, 64, 128), semantic_channels=(16, 32, 64, 128), 
                 semantic_expansion_ratio=6, bga_channels=128, out_indices=(0, 1, 2, 3, 4), align_corners=False, norm_cfg={'type': 'SyncBatchNorm'}, 
                 act_cfg={'type': 'ReLU', 'inplace': True}, pretrained=False, pretrained_model_path=''):
        super(BiSeNetV2, self).__init__()
        # set attributes
        self.structure_type = structure_type
        self.in_channels = in_channels
        self.out_indices = out_indices
        self.detail_channels = detail_channels
        self.semantic_channels = semantic_channels
        self.semantic_expansion_ratio = semantic_expansion_ratio
        self.bga_channels = bga_channels
        self.align_corners = align_corners
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.pretrained = pretrained
        self.pretrained_model_path = pretrained_model_path
        # assert
        if structure_type in AUTO_ASSERT_STRUCTURE_TYPES:
            for key, value in AUTO_ASSERT_STRUCTURE_TYPES[structure_type].items():
                assert hasattr(self, key) and (getattr(self, key) == value)
        # set modules
        self.detail = DetailBranch(self.detail_channels, self.in_channels, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.semantic = SemanticBranch(self.semantic_channels, self.in_channels, self.semantic_expansion_ratio, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.bga = BGALayer(self.bga_channels, self.align_corners, norm_cfg=norm_cfg, act_cfg=act_cfg)
        # load pretrained weights
        if pretrained:
            state_dict = loadpretrainedweights(
                structure_type=structure_type, pretrained_model_path=pretrained_model_path, default_model_urls=DEFAULT_MODEL_URLS
            )
            self.load_state_dict(state_dict, strict=False)
    '''forward'''
    def forward(self, x):
        x_detail = self.detail(x)
        x_semantic_lst = self.semantic(x)
        x_head = self.bga(x_detail, x_semantic_lst[-1])
        outs = x_semantic_lst[:-1] + [x_head]
        outs = [outs[i] for i in self.out_indices]
        return tuple(outs)