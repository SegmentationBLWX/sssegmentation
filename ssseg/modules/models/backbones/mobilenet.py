'''
Function:
    Implementation of MobileNet
Author:
    Zhenchao Jin
'''
import os
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from .bricks import makedivisible, BuildNormalization, BuildActivation, AdptivePaddingConv2d, InvertedResidual, InvertedResidualV3


'''DEFAULT_MODEL_URLS'''
DEFAULT_MODEL_URLS = {
    'mobilenetv2': 'https://download.openmmlab.com/mmclassification/v0/mobilenet_v2/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth',
    'mobilenetv3_small': 'https://download.openmmlab.com/pretrain/third_party/mobilenet_v3_small-47085aa1.pth',
    'mobilenetv3_large': 'https://download.openmmlab.com/pretrain/third_party/mobilenet_v3_large-bc2c3fd3.pth',
}
'''AUTO_ASSERT_STRUCTURE_TYPES'''
AUTO_ASSERT_STRUCTURE_TYPES = {}


'''MobileNetV2'''
class MobileNetV2(nn.Module):
    arch_settings = [[1, 16, 1], [6, 24, 2], [6, 32, 3], [6, 64, 4], [6, 96, 3], [6, 160, 3], [6, 320, 1]]
    def __init__(self, structure_type, in_channels=3, widen_factor=1, outstride=8, out_indices=(1, 2, 4, 6), norm_cfg={'type': 'SyncBatchNorm'}, 
                 act_cfg={'type': 'ReLU6', 'inplace': True}, pretrained=True, pretrained_model_path=''):
        super(MobileNetV2, self).__init__()
        # set attributes
        self.out_indices = out_indices
        self.structure_type = structure_type
        self.in_channels = in_channels
        self.widen_factor = widen_factor
        self.outstride = outstride
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.pretrained = pretrained
        self.pretrained_model_path = pretrained_model_path
        # assert
        if structure_type in AUTO_ASSERT_STRUCTURE_TYPES:
            for key, value in AUTO_ASSERT_STRUCTURE_TYPES[structure_type].items():
                assert hasattr(self, key) and (getattr(self, key) == value)
        # parse outstride
        outstride_to_strides_and_dilations = {
            8: ((1, 2, 2, 1, 1, 1, 1), (1, 1, 1, 2, 2, 4, 4)),
            16: ((1, 2, 2, 2, 1, 1, 1), (1, 1, 1, 1, 1, 2, 2)),
            32: ((1, 2, 2, 2, 1, 2, 1), (1, 1, 1, 1, 1, 1, 1)),
        }
        assert outstride in outstride_to_strides_and_dilations, 'unsupport outstride %s in MobileNetV2' % outstride
        stride_list, dilation_list = outstride_to_strides_and_dilations[outstride]
        # conv1
        self.in_channels = makedivisible(32 * widen_factor, 8)
        self.conv1 = nn.Sequential()
        self.conv1.add_module('conv', nn.Conv2d(in_channels, self.in_channels, kernel_size=3, stride=2, padding=1, bias=False))
        self.conv1.add_module('bn', BuildNormalization(placeholder=self.in_channels, norm_cfg=norm_cfg))
        self.conv1.add_module('activation', BuildActivation(act_cfg))
        # make layers
        self.layers = []
        for i, layer_cfg in enumerate(self.arch_settings):
            expand_ratio, channel, num_blocks = layer_cfg
            stride = stride_list[i]
            dilation = dilation_list[i]
            out_channels = makedivisible(channel * widen_factor, 8)
            inverted_res_layer = self.makelayer(out_channels, num_blocks, stride, dilation, expand_ratio, norm_cfg, act_cfg)
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, inverted_res_layer)
            self.layers.append(layer_name)
        # load pretrained weights
        if pretrained:
            self.loadpretrainedweights(structure_type, pretrained_model_path)
    '''loadpretrainedweights'''
    def loadpretrainedweights(self, structure_type='mobilenetv2', pretrained_model_path=''):
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            checkpoint = torch.load(pretrained_model_path, map_location='cpu')
        else:
            checkpoint = model_zoo.load_url(DEFAULT_MODEL_URLS[structure_type], map_location='cpu')
        if 'state_dict' in checkpoint: 
            state_dict = checkpoint['state_dict']
        else: 
            state_dict = checkpoint
        keys = list(state_dict.keys())
        for key in keys:
            if key.startswith('backbone.'):
                value = state_dict.pop(key)
                key = '.'.join(key.split('.')[1:])
                state_dict[key] = value
        self.load_state_dict(state_dict, strict=False)
    '''forward'''
    def forward(self, x):
        x = self.conv1(x)
        outs = []
        for idx, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if idx in self.out_indices:
                outs.append(x)
        return tuple(outs)
    '''makelayer'''
    def makelayer(self, out_channels, num_blocks, stride, dilation, expand_ratio, norm_cfg=None, act_cfg=None):
        if act_cfg is None: act_cfg = {'type': 'ReLU6', 'inplace': True}
        layers = []
        for i in range(num_blocks):
            layers.append(
                InvertedResidual(
                    self.in_channels,  out_channels, stride=stride if i == 0 else 1, expand_ratio=expand_ratio, 
                    dilation=dilation if i == 0 else 1, norm_cfg=norm_cfg, act_cfg=act_cfg
                )
            )
            self.in_channels = out_channels
        return nn.Sequential(*layers)


'''MobileNetV3'''
class MobileNetV3(nn.Module):
    arch_settings = {
        'small': [
            [3, 16, 16, True, {'type': 'ReLU'}, 2], [3, 72, 24, False, {'type': 'ReLU'}, 2], [3, 88, 24, False, {'type': 'ReLU'}, 1],
            [5, 96, 40, True, {'type': 'HardSwish'}, 2], [5, 240, 40, True, {'type': 'HardSwish'}, 1], [5, 240, 40, True, {'type': 'HardSwish'}, 1],
            [5, 120, 48, True, {'type': 'HardSwish'}, 1], [5, 144, 48, True, {'type': 'HardSwish'}, 1], [5, 288, 96, True, {'type': 'HardSwish'}, 2],
            [5, 576, 96, True, {'type': 'HardSwish'}, 1], [5, 576, 96, True, {'type': 'HardSwish'}, 1],
        ],
        'large': [
            [3, 16, 16, False, {'type': 'ReLU'}, 1], [3, 64, 24, False, {'type': 'ReLU'}, 2], [3, 72, 24, False, {'type': 'ReLU'}, 1],
            [5, 72, 40, True, {'type': 'ReLU'}, 2], [5, 120, 40, True, {'type': 'ReLU'}, 1], [5, 120, 40, True, {'type': 'ReLU'}, 1],
            [3, 240, 80, False, {'type': 'HardSwish'}, 2], [3, 200, 80, False, {'type': 'HardSwish'}, 1], [3, 184, 80, False, {'type': 'HardSwish'}, 1],
            [3, 184, 80, False, {'type': 'HardSwish'}, 1], [3, 480, 112, True, {'type': 'HardSwish'}, 1], [3, 672, 112, True, {'type': 'HardSwish'}, 1],
            [5, 672, 160, True, {'type': 'HardSwish'}, 2], [5, 960, 160, True, {'type': 'HardSwish'}, 1], [5, 960, 160, True, {'type': 'HardSwish'}, 1],
        ],
    }
    def __init__(self, structure_type, in_channels=3, arch_type='large', outstride=8, out_indices=(1, 3, 16), reduction_factor=1, norm_cfg={'type': 'SyncBatchNorm'}, 
                 act_cfg={'type': 'HardSwish'}, pretrained=True, pretrained_model_path=''):
        super(MobileNetV3, self).__init__()
        # set attributes
        self.structure_type = structure_type
        self.in_channels = in_channels
        self.arch_type = arch_type
        self.outstride = outstride
        self.out_indices = out_indices
        self.reduction_factor = reduction_factor
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.pretrained = pretrained
        self.pretrained_model_path = pretrained_model_path
        # assert
        assert arch_type in self.arch_settings
        assert isinstance(reduction_factor, int) and reduction_factor > 0
        assert outstride in [8, 16, 32], 'unsupport outstride %s in MobileNetV3' % outstride
        if structure_type in AUTO_ASSERT_STRUCTURE_TYPES:
            for key, value in AUTO_ASSERT_STRUCTURE_TYPES[structure_type].items():
                assert hasattr(self, key) and (getattr(self, key) == value)
        # set layers
        self.layers = self.makelayers(in_channels, arch_type, reduction_factor, outstride, norm_cfg, act_cfg)
        # load pretrained weights
        if pretrained:
            self.loadpretrainedweights(structure_type, pretrained_model_path)
    '''loadpretrainedweights'''
    def loadpretrainedweights(self, structure_type='mobilenetv3_small', pretrained_model_path=''):
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            checkpoint = torch.load(pretrained_model_path, map_location='cpu')
        else:
            checkpoint = model_zoo.load_url(DEFAULT_MODEL_URLS[structure_type], map_location='cpu')
        if 'state_dict' in checkpoint: 
            state_dict = checkpoint['state_dict']
        else: 
            state_dict = checkpoint
        keys = list(state_dict.keys())
        for key in keys:
            if key.startswith('backbone.'):
                value = state_dict.pop(key)
                key = '.'.join(key.split('.')[1:])
                state_dict[key] = value
        self.load_state_dict(state_dict, strict=False)
    '''make layers'''
    def makelayers(self, in_channels, arch_type, reduction_factor, outstride, norm_cfg=None, act_cfg=None):
        layers, act_cfg_default = [], act_cfg.copy()
        # build the first layer
        in_channels_first_layer, in_channels = in_channels, 16
        layer = nn.Sequential()
        layer.add_module('conv', AdptivePaddingConv2d(in_channels_first_layer, in_channels, kernel_size=3, stride=2, padding=1, bias=False))
        layer.add_module('bn', BuildNormalization(placeholder=in_channels, norm_cfg=norm_cfg))
        layer.add_module('activation', BuildActivation(act_cfg_default))
        self.add_module('layer0', layer)
        layers.append('layer0')
        # build the middle layers
        layer_setting = self.arch_settings[arch_type]
        for i, params in enumerate(layer_setting):
            (kernel_size, mid_channels, out_channels, with_se, act_cfg, stride) = params
            if (arch_type == 'large' and i >= 12) or (arch_type == 'small' and i >= 8):
                mid_channels = mid_channels // reduction_factor
                out_channels = out_channels // reduction_factor
            se_cfg = None
            if with_se:
                se_cfg = {
                    'channels': mid_channels, 'ratio': 4, 'act_cfgs': ({'type': 'ReLU'}, {'type': 'HardSigmoid', 'bias': 3.0, 'divisor': 6.0})
                }
            layer = InvertedResidualV3(
                in_channels=in_channels, out_channels=out_channels, mid_channels=mid_channels, kernel_size=kernel_size,
                stride=stride, se_cfg=se_cfg, with_expand_conv=(in_channels != mid_channels), norm_cfg=norm_cfg, act_cfg=act_cfg,
            )
            in_channels = out_channels
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, layer)
            layers.append(layer_name)
        # build the last layer
        out_channels = 576 if arch_type == 'small' else 960
        layer = nn.Sequential()
        layer.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, dilation={8: 4, 16: 2, 32: 1}[outstride], padding=0, bias=False))
        layer.add_module('bn', BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg))
        layer.add_module('activation', BuildActivation(act_cfg_default))
        layer_name = 'layer{}'.format(len(layer_setting) + 1)
        self.add_module(layer_name, layer)
        layers.append(layer_name)
        # convert backbone MobileNetV3 to a semantic segmentation version
        if outstride == 32: return layers
        if arch_type == 'small':
            self.layer4.depthwise_conv[0].stride = (1, 1)
            if outstride == 8:
                self.layer9.depthwise_conv[0].stride = (1, 1)
            for i in range(4, len(layers)):
                layer = getattr(self, layers[i])
                if isinstance(layer, InvertedResidualV3): modified_module = layer.depthwise_conv[0]
                else: modified_module = layer[0]
                if i < 9 or (outstride == 16):
                    modified_module.dilation = (2, 2)
                    pad = 2
                else:
                    modified_module.dilation = (4, 4)
                    pad = 4
                if not isinstance(modified_module, AdptivePaddingConv2d):
                    pad *= (modified_module.kernel_size[0] - 1) // 2
                    modified_module.padding = (pad, pad)
        else:
            self.layer7.depthwise_conv[0].stride = (1, 1)
            if outstride == 8:
                self.layer13.depthwise_conv[0].stride = (1, 1)
            for i in range(7, len(layers)):
                layer = getattr(self, layers[i])
                if isinstance(layer, InvertedResidualV3): modified_module = layer.depthwise_conv[0]
                else: modified_module = layer[0]
                if i < 13 or (outstride == 16):
                    modified_module.dilation = (2, 2)
                    pad = 2
                else:
                    modified_module.dilation = (4, 4)
                    pad = 4
                if not isinstance(modified_module, AdptivePaddingConv2d):
                    pad *= (modified_module.kernel_size[0] - 1) // 2
                    modified_module.padding = (pad, pad)
        # return layers
        return layers
    '''forward'''
    def forward(self, x):
        outs = []
        for idx, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if idx in self.out_indices:
                outs.append(x)
        return tuple(outs)