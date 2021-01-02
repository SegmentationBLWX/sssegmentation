'''
Function:
    Implementation of ResNeSt
Author:
    Zhenchao Jin
'''
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from .resnet import ResNet
from .resnet import Bottleneck as _Bottleneck
from .bricks import BuildNormalization, BuildActivation


'''model urls'''
model_urls = {
    'resnest50': 'https://download.openmmlab.com/pretrain/third_party/resnest50_d2-7497a55b.pth',
    'resnest101': 'https://download.openmmlab.com/pretrain/third_party/resnest101_d2-f3b931b2.pth',
    'resnest200': 'https://download.openmmlab.com/pretrain/third_party/resnest200_d2-ca88e41f.pth',
}


'''Radix Softmax module'''
class RSoftmax(nn.Module):
    def __init__(self, radix, groups):
        super(RSoftmax, self).__init__()
        self.radix = radix
        self.groups = groups
    '''forward'''
    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.groups, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


'''Split-Attention Conv2d in ResNeSt'''
class SplitAttentionConv2d(nn.Module):
    def __init__(self, in_channels, channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, radix=2, reduction_factor=4, norm_cfg=None, act_cfg=None, **kwargs):
        super(SplitAttentionConv2d, self).__init__()
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.conv = nn.Conv2d(in_channels, channels * radix, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups * radix, bias=False)
        self.bn0 = BuildNormalization(norm_cfg['type'], (channels * radix, norm_cfg['opts']))
        self.relu = BuildActivation(act_cfg['type'], **act_cfg['opts'])
        self.fc1 = nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0, groups=groups)
        self.bn1 = BuildNormalization(norm_cfg['type'], (inter_channels, norm_cfg['opts']))
        self.fc2 = nn.Conv2d(inter_channels, channels * radix, kernel_size=1, stride=1, padding=0, groups=groups)
        self.rsoftmax = RSoftmax(radix, groups)
    '''forward'''
    def forward(self, x):
        x = self.conv(x)
        x = self.bn0(x)
        x = self.relu(x)
        batch, rchannel = x.shape[:2]
        batch = x.size(0)
        if self.radix > 1:
            splits = x.view(batch, self.radix, -1, *x.shape[2:])
            gap = splits.sum(dim=1)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)
        gap = self.bn1(gap)
        gap = self.relu(gap)
        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)
        if self.radix > 1:
            attens = atten.view(batch, self.radix, -1, *atten.shape[2:])
            out = torch.sum(attens * splits, dim=1)
        else:
            out = atten * x
        return out.contiguous()


'''Bottleneck block for ResNeSt'''
class Bottleneck(_Bottleneck):
    expansion = 4
    def __init__(self, inplanes, planes, groups=1, base_width=4, base_channels=64, radix=2, reduction_factor=4, use_avg_after_block_conv2=True, **kwargs):
        super(Bottleneck, self).__init__(inplanes, planes, **kwargs)
        if groups == 1: width = planes
        else: width = math.floor(planes * (base_width / base_channels)) * groups
        norm_cfg, act_cfg = kwargs['norm_cfg'], kwargs['act_cfg']
        self.use_avg_after_block_conv2 = use_avg_after_block_conv2 and self.stride > 1
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = BuildNormalization(norm_cfg['type'], (width, norm_cfg['opts']))
        self.conv2 = SplitAttentionConv2d(
            in_channels=width, 
            channels=width, 
            kernel_size=3, 
            stride=1 if self.use_avg_after_block_conv2 else self.stride, 
            padding=kwargs['dilation'],
            dilation=kwargs['dilation'],
            groups=groups,
            radix=radix,
            reduction_factor=reduction_factor,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        delattr(self, 'bn2')
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = BuildNormalization(norm_cfg['type'], (planes * self.expansion, norm_cfg['opts']))
        if self.use_avg_after_block_conv2: self.avd_layer = nn.AvgPool2d(3, self.stride, padding=1)
    '''forward'''
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.use_avg_after_block_conv2: out = self.avd_layer(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None: identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


'''ResNeSt'''
class ResNeSt(ResNet):
    arch_settings = {
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
        200: (Bottleneck, (3, 24, 36, 3))
    }
    def __init__(self, groups=1, base_width=4, radix=2, reduction_factor=4, use_avg_after_block_conv2=True, **kwargs):
        self.extra_args_for_makelayer = {
            'radix': radix,
            'groups': groups,
            'base_width': base_width,
            'reduction_factor': reduction_factor,
            'base_channels': kwargs['base_channels'],
            'use_avg_after_block_conv2': use_avg_after_block_conv2,
        }
        super(ResNeSt, self).__init__(**kwargs)
    '''make res layer'''
    def makelayer(self, block, inplanes, planes, num_blocks, stride=1, dilation=1, contract_dilation=True, use_avg_for_downsample=False, norm_cfg=None, act_cfg=None, **kwargs):
        kwargs.update(self.extra_args_for_makelayer)
        downsample = None
        dilations = [dilation] * num_blocks
        if contract_dilation and dilation > 1: dilations[0] = dilation // 2
        if stride != 1 or inplanes != planes * block.expansion:
            if use_avg_for_downsample:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
                    nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=1, padding=0, bias=False),
                    BuildNormalization(norm_cfg['type'], (planes * block.expansion, norm_cfg['opts']))
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, padding=0, bias=False),
                    BuildNormalization(norm_cfg['type'], (planes * block.expansion, norm_cfg['opts']))
                )
        layers = []
        layers.append(block(inplanes, planes, stride=stride, dilation=dilations[0], downsample=downsample, norm_cfg=norm_cfg, act_cfg=act_cfg, **kwargs))
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks): layers.append(block(planes * block.expansion, planes, stride=1, dilation=dilations[i], norm_cfg=norm_cfg, act_cfg=act_cfg, **kwargs))
        return nn.Sequential(*layers)


'''build resnest'''
def BuildResNeSt(resnest_type, **kwargs):
    # assert whether support
    supported_resnests = {
        'resnest50': {'depth': 50},
        'resnest101': {'depth': 101},
        'resnest152': {'depth': 152},
        'resnest200': {'depth': 200},
    }
    assert resnest_type in supported_resnests, 'unsupport the resnest_type %s...' % resnest_type
    # parse args
    default_args = {
        'radix': 2,
        'groups': 1,
        'outstride': 8,
        'base_width': 4,
        'use_stem': True,
        'norm_cfg': None,
        'in_channels': 3,
        'pretrained': True,
        'base_channels': 64,
        'stem_channels': 128,
        'reduction_factor': 4,
        'contract_dilation': True,
        'out_indices': (0, 1, 2, 3),
        'pretrained_model_path': '',
        'use_avg_for_downsample': True,
        'use_avg_after_block_conv2': True,
        'act_cfg': {'type': 'relu', 'opts': {'inplace': True}},
    }
    for key, value in kwargs.items():
        if key in default_args: default_args.update({key: value})
    # obtain args for instanced resnest
    resnest_args = supported_resnests[resnest_type]
    resnest_args.update(default_args)
    # obtain the instanced resnest
    model = ResNeSt(**resnest_args)
    # load weights of pretrained model
    if default_args['pretrained'] and os.path.exists(default_args['pretrained_model_path']):
        checkpoint = torch.load(default_args['pretrained_model_path'])
        if 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
        else: state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    elif default_args['pretrained']:
        checkpoint = model_zoo.load_url(model_urls[resnest_type])
        if 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
        else: state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    # return the model
    return model