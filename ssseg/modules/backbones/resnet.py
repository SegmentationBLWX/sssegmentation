'''
Function:
    define the resnet
Author:
    Zhenchao Jin
'''
import os
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from .layers import BuildNormalizationLayer


'''model urls'''
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnet50impro': 'https://openmmlab.oss-accelerate.aliyuncs.com/pretrain/third_party/resnet50_v1c-2cccc1ad.pth',
    'resnet101impro': 'https://openmmlab.oss-accelerate.aliyuncs.com/pretrain/third_party/resnet101_v1c-e67eebb6.pth',
}


'''basic block'''
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, normlayer_opts=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = BuildNormalizationLayer(normlayer_opts['type'], (planes, normlayer_opts['opts']))
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BuildNormalizationLayer(normlayer_opts['type'], (planes, normlayer_opts['opts']))
        self.relu = nn.ReLU(inplace=True)
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


'''bottleneck'''
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, normlayer_opts=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = BuildNormalizationLayer(normlayer_opts['type'], (planes, normlayer_opts['opts']))
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = BuildNormalizationLayer(normlayer_opts['type'], (planes, normlayer_opts['opts']))
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = BuildNormalizationLayer(normlayer_opts['type'], (planes * self.expansion, normlayer_opts['opts']))
        self.relu = nn.ReLU(inplace=True)
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


'''resnet'''
class ResNet(nn.Module):
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }
    def __init__(self, depth, outstride, normlayer_opts, contract_dilation=True, is_improved_version=True, out_indices=(0, 1, 2, 3), **kwargs):
        super(ResNet, self).__init__()
        self.inplanes = 64
        # set out_indices
        self.out_indices = out_indices
        # parse depth settings
        assert depth in self.arch_settings, 'unsupport depth %s in ResNet...' % depth
        block, num_blocks_list = self.arch_settings[depth]
        # parse outstride
        outstride_to_strides_and_dilations = {
            8: ((1, 2, 1, 1), (1, 1, 2, 4)),
            16: ((1, 2, 2, 1), (1, 1, 1, 2)),
            32: ((1, 2, 2, 2), (1, 1, 1, 1)),
        }
        assert outstride in outstride_to_strides_and_dilations, 'unsupport outstride %s in ResNet...' % outstride
        stride_list, dilation_list = outstride_to_strides_and_dilations[outstride]
        # whether replace the 7x7 conv in the input stem with three 3x3 convs
        self.is_improved_version = is_improved_version
        if is_improved_version:
            self.stem = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
                                      BuildNormalizationLayer(normlayer_opts['type'], (32, normlayer_opts['opts'])),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
                                      BuildNormalizationLayer(normlayer_opts['type'], (32, normlayer_opts['opts'])),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                      BuildNormalizationLayer(normlayer_opts['type'], (64, normlayer_opts['opts'])),
                                      nn.ReLU(inplace=True))
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = BuildNormalizationLayer(normlayer_opts['type'], (64, normlayer_opts['opts']))
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # make layers
        self.layer1 = self.makelayer(block, 64, 64, num_blocks_list[0], stride=stride_list[0], dilation=dilation_list[0], normlayer_opts=normlayer_opts, contract_dilation=contract_dilation)
        self.layer2 = self.makelayer(block, 256, 128, num_blocks_list[1], stride=stride_list[1], dilation=dilation_list[1], normlayer_opts=normlayer_opts, contract_dilation=contract_dilation)
        self.layer3 = self.makelayer(block, 512, 256, num_blocks_list[2], stride=stride_list[2], dilation=dilation_list[2], normlayer_opts=normlayer_opts, contract_dilation=contract_dilation)
        self.layer4 = self.makelayer(block, 1024, 512, num_blocks_list[3], stride=stride_list[3], dilation=dilation_list[3], normlayer_opts=normlayer_opts, contract_dilation=contract_dilation)
    '''make res layer'''
    def makelayer(self, block, inplanes, planes, num_blocks, stride=1, dilation=1, normlayer_opts=None, contract_dilation=True):
        downsample = None
        dilations = [dilation] * num_blocks
        if contract_dilation and dilation > 1: dilations[0] = dilation // 2
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, padding=0, bias=False),
                                       BuildNormalizationLayer(normlayer_opts['type'], (planes * block.expansion, normlayer_opts['opts'])))
        layers = []
        layers.append(block(inplanes, planes, stride=stride, dilation=dilations[0], downsample=downsample, normlayer_opts=normlayer_opts))
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks): layers.append(block(planes * block.expansion, planes, stride=1, dilation=dilations[i], normlayer_opts=normlayer_opts))
        return nn.Sequential(*layers)
    '''forward'''
    def forward(self, x):
        if self.is_improved_version:
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
        if len(outs) == 1: return outs[0]
        else: return tuple(outs)


'''build resnet'''
def BuildResNet(resnet_type, **kwargs):
    # assert whether support
    supported_resnets = {
        'resnet18': {'depth': 18},
        'resnet34': {'depth': 34},
        'resnet50': {'depth': 50},
        'resnet101': {'depth': 101},
        'resnet152': {'depth': 152},
    }
    assert resnet_type in supported_resnets, 'unsupport the resnet_type %s...' % resnet_type
    # parse args
    outstride = kwargs.get('outstride', 8)
    pretrained = kwargs.get('pretrained', True)
    normlayer_opts = kwargs.get('normlayer_opts', None)
    contract_dilation = kwargs.get('contract_dilation', True)
    pretrained_model_path = kwargs.get('pretrained_model_path', '')
    is_improved_version = kwargs.get('is_improved_version', True)
    out_indices = kwargs.get('out_indices', (0, 1, 2, 3))
    # obtain args for instanced resnet
    resnet_args = supported_resnets[resnet_type]
    resnet_args.update({
        'outstride': outstride,
        'normlayer_opts': normlayer_opts,
        'contract_dilation': contract_dilation,
        'is_improved_version': is_improved_version,
        'out_indices': out_indices,
    })
    # obtain the instanced resnet
    model = ResNet(**resnet_args)
    # load weights of pretrained model
    if is_improved_version: resnet_type = resnet_type + 'impro'
    if pretrained and os.path.exists(pretrained_model_path):
        checkpoint = torch.load(pretrained_model_path)
        if 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
        else: state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    elif pretrained:
        checkpoint = model_zoo.load_url(model_urls[resnet_type])
        if 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
        else: state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    # return the model
    return model