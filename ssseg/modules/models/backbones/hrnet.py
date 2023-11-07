'''
Function:
    Implementation of HRNet
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...utils import loadpretrainedweights
from .resnet import BasicBlock, Bottleneck
from .bricks import BuildNormalization, BuildActivation


'''DEFAULT_MODEL_URLS'''
DEFAULT_MODEL_URLS = {
    'hrnetv2_w18_small': 'https://download.openmmlab.com/pretrain/third_party/hrnetv2_w18_small-b5a04e21.pth',
    'hrnetv2_w18': 'https://download.openmmlab.com/pretrain/third_party/hrnetv2_w18-00eb2006.pth',
    'hrnetv2_w32': 'https://download.openmmlab.com/pretrain/third_party/hrnetv2_w32-dc9eeb4f.pth',
    'hrnetv2_w40': 'https://download.openmmlab.com/pretrain/third_party/hrnetv2_w40-ed0b031c.pth',
    'hrnetv2_w48': 'https://download.openmmlab.com/pretrain/third_party/hrnetv2_w48-d2186c55.pth',
}
'''AUTO_ASSERT_STRUCTURE_TYPES'''
AUTO_ASSERT_STRUCTURE_TYPES = {}


'''HRModule'''
class HRModule(nn.Module):
    def __init__(self, num_branches, block, num_blocks, in_channels, num_channels, multiscale_output=True, norm_cfg=None, act_cfg=None):
        super(HRModule, self).__init__()
        self.checkbranches(num_branches, num_blocks, in_channels, num_channels)
        self.in_channels = in_channels
        self.num_branches = num_branches
        self.multiscale_output = multiscale_output
        self.branches = self.makebranches(num_branches, block, num_blocks, num_channels, norm_cfg, act_cfg)
        self.fuse_layers = self.makefuselayers(norm_cfg, act_cfg)
        self.relu = BuildActivation(act_cfg)
    '''forward'''
    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = 0
            for j in range(self.num_branches):
                if i == j:
                    y += x[j]
                elif j > i:
                    y = y + F.interpolate(self.fuse_layers[i][j](x[j]), size=x[i].shape[2:], mode='bilinear', align_corners=False)
                else:
                    y += self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse
    '''check branches'''
    def checkbranches(self, num_branches, num_blocks, in_channels, num_channels):
        assert num_branches == len(num_blocks), 'num_branches should be equal to len(num_blocks)'
        assert num_branches == len(num_channels), 'num_branches should be equal to len(num_channels)'
        assert num_branches == len(in_channels), 'num_branches should be equal to len(in_channels)'
    '''make branches'''
    def makebranches(self, num_branches, block, num_blocks, num_channels, norm_cfg=None, act_cfg=None):
        branches = []
        for i in range(num_branches):
            branches.append(self.makebranch(i, block, num_blocks, num_channels, norm_cfg=norm_cfg, act_cfg=act_cfg))
        return nn.ModuleList(branches)
    '''make one branch'''
    def makebranch(self, branch_index, block, num_blocks, num_channels, stride=1, norm_cfg=None, act_cfg=None):
        downsample = None
        if stride != 1 or self.in_channels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels[branch_index], num_channels[branch_index] * block.expansion, kernel_size=1, stride=stride, padding=0, bias=False),
                BuildNormalization(placeholder=num_channels[branch_index] * block.expansion, norm_cfg=norm_cfg),
            )
        layers = []
        layers.append(block(self.in_channels[branch_index], num_channels[branch_index], stride, downsample=downsample, norm_cfg=norm_cfg, act_cfg=act_cfg))
        self.in_channels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.in_channels[branch_index], num_channels[branch_index], norm_cfg=norm_cfg, act_cfg=act_cfg))
        return nn.Sequential(*layers)
    '''make fuse layer'''
    def makefuselayers(self, norm_cfg=None, act_cfg=None):
        if self.num_branches == 1: return None
        num_branches = self.num_branches
        in_channels = self.in_channels
        fuse_layers = []
        num_out_branches = num_branches if self.multiscale_output else 1
        for i in range(num_out_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(in_channels[j], in_channels[i], kernel_size=1, stride=1, padding=0, bias=False),
                        BuildNormalization(placeholder=in_channels[i], norm_cfg=norm_cfg),
                        nn.Upsample(scale_factor=2**(j-i), mode='bilinear', align_corners=False)
                    ))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(nn.Sequential(
                                nn.Conv2d(in_channels[j], in_channels[i], kernel_size=3, stride=2, padding=1, bias=False),
                                BuildNormalization(placeholder=in_channels[i], norm_cfg=norm_cfg),
                            ))
                        else:
                            conv_downsamples.append(nn.Sequential(
                                nn.Conv2d(in_channels[j], in_channels[j], kernel_size=3, stride=2, padding=1, bias=False),
                                BuildNormalization(placeholder=in_channels[j], norm_cfg=norm_cfg),
                                BuildActivation(act_cfg),
                            ))
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)


'''HRNet'''
class HRNet(nn.Module):
    blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}
    arch_settings = {
        'hrnetv2_w18_small': {
            'stage1': {'num_modules': 1, 'num_branches': 1, 'block': 'BOTTLENECK', 'num_blocks': (2,), 'num_channels': (64,),},
            'stage2': {'num_modules': 1, 'num_branches': 2, 'block': 'BASIC', 'num_blocks': (2, 2), 'num_channels': (18, 36),},
            'stage3': {'num_modules': 3, 'num_branches': 3, 'block': 'BASIC', 'num_blocks': (2, 2, 2), 'num_channels': (18, 36, 72),},
            'stage4': {'num_modules': 2, 'num_branches': 4, 'block': 'BASIC', 'num_blocks': (2, 2, 2, 2), 'num_channels': (18, 36, 72, 144),},
        },
        'hrnetv2_w18': {
            'stage1': {'num_modules': 1, 'num_branches': 1, 'block': 'BOTTLENECK', 'num_blocks': (4,), 'num_channels': (64,),},
            'stage2': {'num_modules': 1, 'num_branches': 2, 'block': 'BASIC', 'num_blocks': (4, 4), 'num_channels': (18, 36),},
            'stage3': {'num_modules': 4, 'num_branches': 3, 'block': 'BASIC', 'num_blocks': (4, 4, 4), 'num_channels': (18, 36, 72),},
            'stage4': {'num_modules': 3, 'num_branches': 4, 'block': 'BASIC', 'num_blocks': (4, 4, 4, 4), 'num_channels': (18, 36, 72, 144),},
        },
        'hrnetv2_w32': {
            'stage1': {'num_modules': 1, 'num_branches': 1, 'block': 'BOTTLENECK', 'num_blocks': (4,), 'num_channels': (64,),},
            'stage2': {'num_modules': 1, 'num_branches': 2, 'block': 'BASIC', 'num_blocks': (4, 4), 'num_channels': (32, 64),},
            'stage3': {'num_modules': 4, 'num_branches': 3, 'block': 'BASIC', 'num_blocks': (4, 4, 4), 'num_channels': (32, 64, 128),},
            'stage4': {'num_modules': 3, 'num_branches': 4, 'block': 'BASIC', 'num_blocks': (4, 4, 4, 4), 'num_channels': (32, 64, 128, 256),},
        },
        'hrnetv2_w40': {
            'stage1': {'num_modules': 1, 'num_branches': 1, 'block': 'BOTTLENECK', 'num_blocks': (4,), 'num_channels': (64,),},
            'stage2': {'num_modules': 1, 'num_branches': 2, 'block': 'BASIC', 'num_blocks': (4, 4), 'num_channels': (40, 80),},
            'stage3': {'num_modules': 4, 'num_branches': 3, 'block': 'BASIC', 'num_blocks': (4, 4, 4), 'num_channels': (40, 80, 160),},
            'stage4': {'num_modules': 3, 'num_branches': 4, 'block': 'BASIC', 'num_blocks': (4, 4, 4, 4), 'num_channels': (40, 80, 160, 320),},
        },
        'hrnetv2_w48': {
            'stage1': {'num_modules': 1, 'num_branches': 1, 'block': 'BOTTLENECK', 'num_blocks': (4,), 'num_channels': (64,),},
            'stage2': {'num_modules': 1, 'num_branches': 2, 'block': 'BASIC', 'num_blocks': (4, 4), 'num_channels': (48, 96),},
            'stage3': {'num_modules': 4, 'num_branches': 3, 'block': 'BASIC', 'num_blocks': (4, 4, 4), 'num_channels': (48, 96, 192),},
            'stage4': {'num_modules': 3, 'num_branches': 4, 'block': 'BASIC', 'num_blocks': (4, 4, 4, 4), 'num_channels': (48, 96, 192, 384),},
        },
    }
    def __init__(self, structure_type, arch='hrnetv2_w18_small', in_channels=3, norm_cfg={'type': 'SyncBatchNorm'}, 
                 act_cfg={'type': 'ReLU', 'inplace': True}, pretrained=True, pretrained_model_path=''):
        super(HRNet, self).__init__()
        # set attributes
        self.structure_type = structure_type
        self.arch = arch
        self.in_channels = in_channels
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.pretrained = pretrained
        self.pretrained_model_path = pretrained_model_path
        # assert
        if structure_type in AUTO_ASSERT_STRUCTURE_TYPES:
            for key, value in AUTO_ASSERT_STRUCTURE_TYPES[structure_type].items():
                assert hasattr(self, key) and (getattr(self, key) == value)
        # stem net
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = BuildNormalization(placeholder=64, norm_cfg=norm_cfg)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = BuildNormalization(placeholder=64, norm_cfg=norm_cfg)
        self.relu = BuildActivation(act_cfg)
        # stage1
        stages_cfg = self.arch_settings[arch]
        self.stage1_cfg = stages_cfg['stage1']
        num_channels = self.stage1_cfg['num_channels'][0]
        block_type = self.stage1_cfg['block']
        num_blocks = self.stage1_cfg['num_blocks'][0]
        block = self.blocks_dict[block_type]
        stage1_out_channels = num_channels * block.expansion
        self.layer1 = self.makelayer(block, 64, num_channels, num_blocks, norm_cfg=norm_cfg, act_cfg=act_cfg)
        # stage2
        self.stage2_cfg = stages_cfg['stage2']
        num_channels = self.stage2_cfg['num_channels']
        block_type = self.stage2_cfg['block']
        block = self.blocks_dict[block_type]
        num_channels = [channel * block.expansion for channel in num_channels]
        self.transition1 = self.maketransitionlayer([stage1_out_channels], num_channels, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.stage2, pre_stage_channels = self.makestage(self.stage2_cfg, num_channels, norm_cfg=norm_cfg, act_cfg=act_cfg)
        # stage3
        self.stage3_cfg = stages_cfg['stage3']
        num_channels = self.stage3_cfg['num_channels']
        block_type = self.stage3_cfg['block']
        block = self.blocks_dict[block_type]
        num_channels = [channel * block.expansion for channel in num_channels]
        self.transition2 = self.maketransitionlayer(pre_stage_channels, num_channels, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.stage3, pre_stage_channels = self.makestage(self.stage3_cfg, num_channels, norm_cfg=norm_cfg, act_cfg=act_cfg)
        # stage4
        self.stage4_cfg = stages_cfg['stage4']
        num_channels = self.stage4_cfg['num_channels']
        block_type = self.stage4_cfg['block']
        block = self.blocks_dict[block_type]
        num_channels = [channel * block.expansion for channel in num_channels]
        self.transition3 = self.maketransitionlayer(pre_stage_channels, num_channels, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.stage4, pre_stage_channels = self.makestage(self.stage4_cfg, num_channels, norm_cfg=norm_cfg, act_cfg=act_cfg)
        # load pretrained weights
        if pretrained:
            state_dict = loadpretrainedweights(
                structure_type=structure_type, pretrained_model_path=pretrained_model_path, default_model_urls=DEFAULT_MODEL_URLS
            )
            self.load_state_dict(state_dict, strict=False)
    '''forward'''
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)
        x_list = []
        for i in range(self.stage2_cfg['num_branches']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        x_list = []
        for i in range(self.stage3_cfg['num_branches']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        x_list = []
        for i in range(self.stage4_cfg['num_branches']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        h, w = max([y.shape[2] for y in y_list]), max([y.shape[3] for y in y_list])
        out = torch.cat([F.interpolate(y, size=(h, w), mode='bilinear', align_corners=False) for y in y_list], dim=1)
        outs = [out]
        return tuple(outs)
    '''make stage'''
    def makestage(self, layer_config, in_channels, multiscale_output=True, norm_cfg=None, act_cfg=None):
        num_modules = layer_config['num_modules']
        num_branches = layer_config['num_branches']
        num_blocks = layer_config['num_blocks']
        num_channels = layer_config['num_channels']
        block = self.blocks_dict[layer_config['block']]
        hr_modules = []
        for i in range(num_modules):
            if not multiscale_output and i == num_modules - 1:
                reset_multiscale_output = False
            else:
                reset_multiscale_output = True
            hr_modules.append(HRModule(num_branches, block, num_blocks, in_channels, num_channels, reset_multiscale_output, norm_cfg, act_cfg))
        return nn.Sequential(*hr_modules), in_channels
    '''make layer'''
    def makelayer(self, block, inplanes, planes, num_blocks, stride=1, norm_cfg=None, act_cfg=None):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=1, padding=0, bias=False),
                BuildNormalization(placeholder=planes * block.expansion, norm_cfg=norm_cfg),
            )
        layers = []
        layers.append(
            block(inplanes, planes, stride, downsample=downsample, norm_cfg=norm_cfg, act_cfg=act_cfg)
        )
        inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(
                block(inplanes, planes, norm_cfg=norm_cfg, act_cfg=act_cfg)
            )
        return nn.Sequential(*layers)
    '''make transition layer'''
    def maketransitionlayer(self, num_channels_pre_layer, num_channels_cur_layer, norm_cfg=None, act_cfg=None):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], kernel_size=3, stride=1, padding=1, bias=False),
                        BuildNormalization(placeholder=num_channels_cur_layer[i], norm_cfg=norm_cfg),
                        BuildActivation(act_cfg),
                    ))
                else:
                    transition_layers.append(None)
            else:
                conv_downsamples = []
                for j in range(i + 1 - num_branches_pre):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = num_channels_cur_layer[i] if j == i - num_branches_pre else in_channels
                    conv_downsamples.append(nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                        BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
                        BuildActivation(act_cfg),
                    ))
                transition_layers.append(nn.Sequential(*conv_downsamples))
        return nn.ModuleList(transition_layers)