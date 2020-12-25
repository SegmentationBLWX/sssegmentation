'''
Function:
    Implementation of MobileNet
Author:
    Zhenchao Jin
'''
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from .bricks import BuildNormalization, InvertedResidual


'''model urls'''
model_urls = {
    'mobilenetv2': 'https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/imagenet/mobilenet_v2_batch256_20200708-3b2dc3af.pth'
}


'''MobileNetV2'''
class MobileNetV2(nn.Module):
    arch_settings = [[1, 16, 1], [6, 24, 2], [6, 32, 3], [6, 64, 4], [6, 96, 3], [6, 160, 3], [6, 320, 1]]
    def __init__(self, in_channels=3, widen_factor=1, outstride=8, out_indices=(1, 2, 4, 6), norm_cfg=None, **kwargs):
        super(MobileNetV2, self).__init__()
        # set out_indices
        self.out_indices = out_indices
        # parse outstride
        outstride_to_strides_and_dilations = {
            8: ((1, 2, 2, 1, 1, 1, 1), (1, 1, 1, 2, 2, 4, 4)),
            16: ((1, 2, 2, 2, 1, 1, 1), (1, 1, 1, 1, 1, 2, 2)),
            32: ((1, 2, 2, 2, 1, 2, 1), (1, 1, 1, 1, 1, 1, 1)),
        }
        assert outstride in outstride_to_strides_and_dilations, 'unsupport outstride %s in MobileNetV2...' % outstride
        stride_list, dilation_list = outstride_to_strides_and_dilations[outstride]
        # conv1
        self.in_channels = self.makedivisible(32 * widen_factor, 8)
        self.conv1 = nn.Sequential()
        self.conv1.add_module('conv', nn.Conv2d(in_channels, self.in_channels, kernel_size=3, stride=2, padding=1, bias=False))
        self.conv1.add_module('bn', BuildNormalization(norm_cfg['type'], (self.in_channels, norm_cfg['opts'])))
        self.conv1.add_module('activation', nn.ReLU6(inplace=True))
        # make layers
        self.layers = []
        for i, layer_cfg in enumerate(self.arch_settings):
            expand_ratio, channel, num_blocks = layer_cfg
            stride = stride_list[i]
            dilation = dilation_list[i]
            out_channels = self.makedivisible(channel * widen_factor, 8)
            inverted_res_layer = self.makelayer(out_channels, num_blocks, stride, dilation, expand_ratio, norm_cfg)
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, inverted_res_layer)
            self.layers.append(layer_name)
    '''forward'''
    def forward(self, x):
        x = self.conv1(x)
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1: return outs[0]
        else: return tuple(outs)
    '''make divisible'''
    def makedivisible(self, value, divisor, min_value=None, min_ratio=0.9):
        if min_value is None: min_value = divisor
        new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
        if new_value < min_ratio * value: new_value += divisor
        return new_value
    '''make layer'''
    def makelayer(self, out_channels, num_blocks, stride, dilation, expand_ratio, norm_cfg, act_cfg=None):
        if act_cfg is None: act_cfg = {'type': 'relu6', 'opts': {'inplace': True}}
        layers = []
        for i in range(num_blocks):
            layers.append(
                InvertedResidual(
                    self.in_channels, 
                    out_channels, 
                    stride=stride if i == 0 else 1, 
                    expand_ratio=expand_ratio, 
                    dilation=dilation if i == 0 else 1, 
                    norm_cfg=norm_cfg, 
                    act_cfg=act_cfg
                )
            )
            self.in_channels = out_channels
        return nn.Sequential(*layers)


'''build mobilenet'''
def BuildMobileNet(mobilenet_type, **kwargs):
    # assert whether support
    supported_mobilenets = {
        'mobilenetv2': MobileNetV2
    }
    assert mobilenet_type in supported_mobilenets, 'unsupport the mobilenet_type %s...' % mobilenet_type
    # parse args
    default_args = {
        'outstride': 8,
        'norm_cfg': None,
        'in_channels': 3,
        'widen_factor': 1,
        'pretrained': True,
        'out_indices': (1, 2, 4, 6),
        'pretrained_model_path': '',
    }
    for key, value in kwargs.items():
        if key in default_args: default_args.update({key: value})
    # obtain the instanced mobilenet
    mobilenet_args = default_args.copy()
    model = supported_mobilenets[mobilenet_type](**mobilenet_args)
    # load weights of pretrained model
    if default_args['pretrained'] and os.path.exists(default_args['pretrained_model_path']):
        checkpoint = torch.load(default_args['pretrained_model_path'])
        if 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
        else: state_dict = checkpoint
        keys = list(state_dict.keys())
        for key in keys:
            if key.startswith('backbone.'):
                value = state_dict.pop(key)
                key = '.'.join(key.split('.')[1:])
                state_dict[key] = value
        model.load_state_dict(state_dict, strict=False)
    elif default_args['pretrained']:
        checkpoint = model_zoo.load_url(model_urls[mobilenet_type])
        if 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
        else: state_dict = checkpoint
        keys = list(state_dict.keys())
        for key in keys:
            if key.startswith('backbone.'):
                value = state_dict.pop(key)
                key = '.'.join(key.split('.')[1:])
                state_dict[key] = value
        model.load_state_dict(state_dict, strict=False)
    # return the model
    return model