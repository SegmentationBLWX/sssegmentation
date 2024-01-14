'''
Function:
    Implementation of CGNet
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
from ...utils import loadpretrainedweights
from .bricks import BuildNormalization, BuildActivation


'''DEFAULT_MODEL_URLS'''
DEFAULT_MODEL_URLS = {}
'''AUTO_ASSERT_STRUCTURE_TYPES'''
AUTO_ASSERT_STRUCTURE_TYPES = {}


'''GlobalContextExtractor'''
class GlobalContextExtractor(nn.Module):
    def __init__(self, channels, reduction=16):
        super(GlobalContextExtractor, self).__init__()
        assert reduction >= 1 and channels >= reduction
        self.channels = channels
        self.reduction = reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction), 
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels), 
            nn.Sigmoid()
        )
    '''forward'''
    def forward(self, x):
        batch_size, num_channels = x.size()[:2]
        y = self.avg_pool(x).view(batch_size, num_channels)
        y = self.fc(y).view(batch_size, num_channels, 1, 1)
        return x * y


'''ContextGuidedBlock'''
class ContextGuidedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=2, reduction=16, skip_connect=True, downsample=False, norm_cfg=None, act_cfg=None):
        super(ContextGuidedBlock, self).__init__()
        # set attrs
        self.downsample = downsample
        self.skip_connect = skip_connect and not downsample
        channels = out_channels if downsample else out_channels // 2
        if 'type' in act_cfg and act_cfg['type'] == 'PReLU':
            act_cfg['num_parameters'] = channels
        kernel_size = 3 if downsample else 1
        stride = 2 if downsample else 1
        padding = (kernel_size - 1) // 2
        # instance modules
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            BuildNormalization(placeholder=channels, norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        self.f_loc = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False)
        self.f_sur = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, groups=channels, bias=False)
        self.bn = BuildNormalization(placeholder=channels * 2, norm_cfg=norm_cfg)
        self.activate = nn.PReLU(2 * channels)
        if downsample:
            self.bottleneck = nn.Conv2d(2 * channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.f_glo = GlobalContextExtractor(out_channels, reduction)
    '''forward'''
    def forward(self, x):
        out = self.conv1x1(x)
        loc = self.f_loc(out)
        sur = self.f_sur(out)
        joi_feat = torch.cat([loc, sur], 1)
        joi_feat = self.bn(joi_feat)
        joi_feat = self.activate(joi_feat)
        if self.downsample:
            joi_feat = self.bottleneck(joi_feat)
        out = self.f_glo(joi_feat)
        if self.skip_connect:
            return x + out
        return out


'''InputInjection'''
class InputInjection(nn.Module):
    def __init__(self, num_downsamplings):
        super(InputInjection, self).__init__()
        self.pools = nn.ModuleList()
        for _ in range(num_downsamplings):
            self.pools.append(nn.AvgPool2d(3, stride=2, padding=1))
    '''forward'''
    def forward(self, x):
        for pool in self.pools:
            x = pool(x)
        return x


'''CGNet'''
class CGNet(nn.Module):
    def __init__(self, structure_type, in_channels=3, num_channels=(32, 64, 128), num_blocks=(3, 21), dilations=(2, 4), reductions=(8, 16), 
                 norm_cfg={'type': 'SyncBatchNorm'}, act_cfg={'type': 'PReLU'}, pretrained=False, pretrained_model_path=''):
        super(CGNet, self).__init__()
        # set attributes
        self.structure_type = structure_type
        self.in_channels = in_channels
        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.dilations = dilations
        self.reductions = reductions
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.pretrained = pretrained
        self.pretrained_model_path = pretrained_model_path
        if 'type' in self.act_cfg and self.act_cfg['type'] == 'PReLU':
            self.act_cfg['num_parameters'] = num_channels[0]
        # assert
        assert isinstance(num_channels, tuple) and len(num_channels) == 3
        assert isinstance(num_blocks, tuple) and len(num_blocks) == 2
        assert isinstance(dilations, tuple) and len(dilations) == 2
        assert isinstance(reductions, tuple) and len(reductions) == 2
        if structure_type in AUTO_ASSERT_STRUCTURE_TYPES:
            for key, value in AUTO_ASSERT_STRUCTURE_TYPES[structure_type].items():
                assert hasattr(self, key) and (getattr(self, key) == value)
        # stem
        cur_channels = in_channels
        self.stem = nn.ModuleList()
        for i in range(3):
            self.stem.append(nn.Sequential(
                nn.Conv2d(cur_channels, num_channels[0], kernel_size=3, stride=2 if i == 0 else 1, padding=1, bias=False),
                BuildNormalization(placeholder=num_channels[0], norm_cfg=norm_cfg),
                BuildActivation(act_cfg),
            ))
            cur_channels = num_channels[0]
        # down-sample for Input, factor=2
        self.inject_2x = InputInjection(1)
        # down-sample for Input, factor=4
        self.inject_4x = InputInjection(2)
        # norm prelu
        cur_channels += in_channels
        self.norm_prelu_0 = nn.Sequential(
            BuildNormalization(placeholder=cur_channels, norm_cfg=norm_cfg),
            nn.PReLU(cur_channels),
        )
        # stage 1
        self.level1 = nn.ModuleList()
        for i in range(num_blocks[0]):
            self.level1.append(ContextGuidedBlock(
                in_channels=cur_channels if i == 0 else num_channels[1], out_channels=num_channels[1], dilation=dilations[0], 
                reduction=reductions[0], skip_connect=True, downsample=(i == 0), norm_cfg=norm_cfg, act_cfg=act_cfg,
            ))
        cur_channels = 2 * num_channels[1] + in_channels
        self.norm_prelu_1 = nn.Sequential(
            BuildNormalization(placeholder=cur_channels, norm_cfg=norm_cfg),
            nn.PReLU(cur_channels),
        )
        # stage 2
        self.level2 = nn.ModuleList()
        for i in range(num_blocks[1]):
            self.level2.append(ContextGuidedBlock(
                in_channels=cur_channels if i == 0 else num_channels[2], out_channels=num_channels[2], dilation=dilations[1],
                reduction=reductions[1], skip_connect=True, downsample=(i == 0), norm_cfg=norm_cfg, act_cfg=act_cfg,
            ))
        cur_channels = 2 * num_channels[2]
        self.norm_prelu_2 = nn.Sequential(
            BuildNormalization(placeholder=cur_channels, norm_cfg=norm_cfg),
            nn.PReLU(cur_channels),
        )
        # load pretrained weights
        if pretrained:
            state_dict = loadpretrainedweights(
                structure_type=structure_type, pretrained_model_path=pretrained_model_path, default_model_urls=DEFAULT_MODEL_URLS
            )
            self.load_state_dict(state_dict, strict=False)
    '''forward'''
    def forward(self, x):
        output = []
        # stage 0
        inp_2x = self.inject_2x(x)
        inp_4x = self.inject_4x(x)
        for layer in self.stem:
            x = layer(x)
        x = self.norm_prelu_0(torch.cat([x, inp_2x], 1))
        output.append(x)
        # stage 1
        for i, layer in enumerate(self.level1):
            x = layer(x)
            if i == 0: down1 = x
        x = self.norm_prelu_1(torch.cat([x, down1, inp_4x], 1))
        output.append(x)
        # stage 2
        for i, layer in enumerate(self.level2):
            x = layer(x)
            if i == 0: down2 = x
        x = self.norm_prelu_2(torch.cat([down2, x], 1))
        output.append(x)
        # return
        return output