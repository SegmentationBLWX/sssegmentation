'''
Function:
    Implementation of ResNet
Author:
    Zhenchao Jin
'''
import os
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from .bricks import BuildNormalization, BuildActivation, constructnormcfg


'''model urls'''
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnet18stem': 'https://download.openmmlab.com/pretrain/third_party/resnet18_v1c-b5776b93.pth',
    'resnet50stem': 'https://download.openmmlab.com/pretrain/third_party/resnet50_v1c-2cccc1ad.pth',
    'resnet101stem': 'https://download.openmmlab.com/pretrain/third_party/resnet101_v1c-e67eebb6.pth',
}


'''BasicBlock'''
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, norm_cfg=None, act_cfg=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = BuildNormalization(constructnormcfg(placeholder=planes, norm_cfg=norm_cfg))
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BuildNormalization(constructnormcfg(placeholder=planes, norm_cfg=norm_cfg))
        self.relu = BuildActivation(act_cfg)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
    '''forward'''
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None: identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


'''Bottleneck'''
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, norm_cfg=None, act_cfg=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = BuildNormalization(constructnormcfg(placeholder=planes, norm_cfg=norm_cfg))
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = BuildNormalization(constructnormcfg(placeholder=planes, norm_cfg=norm_cfg))
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = BuildNormalization(constructnormcfg(placeholder=planes * self.expansion, norm_cfg=norm_cfg))
        self.relu = BuildActivation(act_cfg)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
    '''forward'''
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None: identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


'''ResNet'''
class ResNet(nn.Module):
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }
    def __init__(self, in_channels=3, base_channels=64, stem_channels=64, depth=101, outstride=8, contract_dilation=True, use_stem=True, 
                 out_indices=(0, 1, 2, 3), use_avg_for_downsample=False, norm_cfg=None, act_cfg=None):
        super(ResNet, self).__init__()
        self.inplanes = stem_channels
        # set out_indices
        self.out_indices = out_indices
        # parse depth settings
        assert depth in self.arch_settings, 'unsupport depth %s' % depth
        block, num_blocks_list = self.arch_settings[depth]
        # parse outstride
        outstride_to_strides_and_dilations = {
            8: ((1, 2, 1, 1), (1, 1, 2, 4)),
            16: ((1, 2, 2, 1), (1, 1, 1, 2)),
            32: ((1, 2, 2, 2), (1, 1, 1, 1)),
        }
        assert outstride in outstride_to_strides_and_dilations, 'unsupport outstride %s' % outstride
        stride_list, dilation_list = outstride_to_strides_and_dilations[outstride]
        # whether replace the 7x7 conv in the input stem with three 3x3 convs
        self.use_stem = use_stem
        if use_stem:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, stem_channels // 2, kernel_size=3, stride=2, padding=1, bias=False),
                BuildNormalization(constructnormcfg(placeholder=stem_channels // 2, norm_cfg=norm_cfg)),
                BuildActivation(act_cfg),
                nn.Conv2d(stem_channels // 2, stem_channels // 2, kernel_size=3, stride=1, padding=1, bias=False),
                BuildNormalization(constructnormcfg(placeholder=stem_channels // 2, norm_cfg=norm_cfg)),
                BuildActivation(act_cfg),
                nn.Conv2d(stem_channels // 2, stem_channels, kernel_size=3, stride=1, padding=1, bias=False),
                BuildNormalization(constructnormcfg(placeholder=stem_channels, norm_cfg=norm_cfg)),
                BuildActivation(act_cfg),
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, stem_channels, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = BuildNormalization(constructnormcfg(placeholder=stem_channels, norm_cfg=norm_cfg))
            self.relu = BuildActivation(act_cfg)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # make layers
        self.layer1 = self.makelayer(
            block=block, 
            inplanes=stem_channels, 
            planes=base_channels, 
            num_blocks=num_blocks_list[0], 
            stride=stride_list[0], 
            dilation=dilation_list[0], 
            contract_dilation=contract_dilation, 
            use_avg_for_downsample=use_avg_for_downsample,
            norm_cfg=norm_cfg, 
            act_cfg=act_cfg,
        )
        self.layer2 = self.makelayer(
            block=block, 
            inplanes=base_channels * 4 if depth >= 50 else base_channels, 
            planes=base_channels * 2, 
            num_blocks=num_blocks_list[1], 
            stride=stride_list[1], 
            dilation=dilation_list[1], 
            contract_dilation=contract_dilation, 
            use_avg_for_downsample=use_avg_for_downsample,
            norm_cfg=norm_cfg, 
            act_cfg=act_cfg,
        )
        self.layer3 = self.makelayer(
            block=block, 
            inplanes=base_channels * 8 if depth >= 50 else base_channels * 2, 
            planes=base_channels * 4, 
            num_blocks=num_blocks_list[2], 
            stride=stride_list[2], 
            dilation=dilation_list[2], 
            contract_dilation=contract_dilation, 
            use_avg_for_downsample=use_avg_for_downsample,
            norm_cfg=norm_cfg, 
            act_cfg=act_cfg,
        )
        self.layer4 = self.makelayer(
            block=block, 
            inplanes=base_channels * 16 if depth >= 50 else base_channels * 4,  
            planes=base_channels * 8, 
            num_blocks=num_blocks_list[3], 
            stride=stride_list[3], 
            dilation=dilation_list[3], 
            contract_dilation=contract_dilation, 
            use_avg_for_downsample=use_avg_for_downsample,
            norm_cfg=norm_cfg, 
            act_cfg=act_cfg,
        )
    '''make res layer'''
    def makelayer(self, block, inplanes, planes, num_blocks, stride=1, dilation=1, contract_dilation=True, use_avg_for_downsample=False, norm_cfg=None, act_cfg=None):
        downsample = None
        dilations = [dilation] * num_blocks
        if contract_dilation and dilation > 1: dilations[0] = dilation // 2
        if stride != 1 or inplanes != planes * block.expansion:
            if use_avg_for_downsample:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
                    nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=1, padding=0, bias=False),
                    BuildNormalization(constructnormcfg(placeholder=planes * block.expansion, norm_cfg=norm_cfg))
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, padding=0, bias=False),
                    BuildNormalization(constructnormcfg(placeholder=planes * block.expansion, norm_cfg=norm_cfg))
                )
        layers = []
        layers.append(block(inplanes, planes, stride=stride, dilation=dilations[0], downsample=downsample, norm_cfg=norm_cfg, act_cfg=act_cfg))
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks): 
            layers.append(block(planes * block.expansion, planes, stride=1, dilation=dilations[i], norm_cfg=norm_cfg, act_cfg=act_cfg))
        return nn.Sequential(*layers)
    '''forward'''
    def forward(self, x):
        if self.use_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        outs = []
        for i, feats in enumerate([x1, x2, x3, x4]):
            if i in self.out_indices: outs.append(feats)
        return tuple(outs)


'''BuildResNet'''
def BuildResNet(resnet_cfg):
    # assert whether support
    resnet_type = resnet_cfg.pop('type')
    supported_resnets = {
        'resnet18': {'depth': 18}, 'resnet34': {'depth': 34}, 'resnet50': {'depth': 50},
        'resnet101': {'depth': 101}, 'resnet152': {'depth': 152},
    }
    assert resnet_type in supported_resnets, 'unsupport the resnet_type %s' % resnet_type
    # parse cfg
    default_cfg = {
        'outstride': 8,
        'use_stem': True,
        'norm_cfg': None,
        'in_channels': 3,
        'pretrained': True,
        'base_channels': 64,
        'stem_channels': 64,
        'contract_dilation': True,
        'out_indices': (0, 1, 2, 3),
        'pretrained_model_path': '',
        'use_avg_for_downsample': False,
        'act_cfg': {'type': 'relu', 'inplace': True},
    }
    for key, value in resnet_cfg.items():
        if key in default_cfg: 
            default_cfg.update({key: value})
    # obtain resnet_cfg
    resnet_cfg = default_cfg.copy()
    resnet_cfg.update(supported_resnets[resnet_type])
    pretrained = resnet_cfg.pop('pretrained')
    pretrained_model_path = resnet_cfg.pop('pretrained_model_path')
    # obtain the instanced resnet
    model = ResNet(**resnet_cfg)
    # load weights of pretrained model
    if resnet_cfg['use_stem']: 
        resnet_type = resnet_type + 'stem'
    if pretrained and os.path.exists(pretrained_model_path):
        checkpoint = torch.load(pretrained_model_path)
        if 'state_dict' in checkpoint: 
            state_dict = checkpoint['state_dict']
        else: 
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    elif pretrained:
        checkpoint = model_zoo.load_url(model_urls[resnet_type])
        if 'state_dict' in checkpoint: 
            state_dict = checkpoint['state_dict']
        else: 
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    # return the model
    return model