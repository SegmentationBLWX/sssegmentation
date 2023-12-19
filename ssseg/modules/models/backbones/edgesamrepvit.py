'''
Function:
    Implementation of EdgeSAMRepViT
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .samvit import LayerNorm2d
from .mobilesamtinyvit import Conv2dBN as _Conv2dBN
from .bricks import makedivisible, SqueezeExcitationConv2d


'''DEFAULT_MODEL_URLS'''
DEFAULT_MODEL_URLS = {}
'''AUTO_ASSERT_STRUCTURE_TYPES'''
AUTO_ASSERT_STRUCTURE_TYPES = {}


'''Conv2dBN'''
class Conv2dBN(_Conv2dBN):
    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = nn.Conv2d(w.size(1) * self.c.groups, w.size(0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups, device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


'''Residual'''
class Residual(nn.Module):
    def __init__(self, m, drop=0.):
        super(Residual, self).__init__()
        self.m = m
        self.drop = drop
    '''forward'''
    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1, device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)
    '''fuse'''
    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, Conv2dBN):
            m = self.m.fuse()
            assert (m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = F.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        elif isinstance(self.m, nn.Conv2d):
            m = self.m
            assert (m.groups != m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = F.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self


'''RepVGGDW'''
class RepVGGDW(nn.Module):
    def __init__(self, ed):
        super(RepVGGDW, self).__init__()
        self.conv = Conv2dBN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = Conv2dBN(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
    '''forward'''
    def forward(self, x):
        return self.conv(x) + self.conv1(x) + x
    '''fuse'''
    @torch.no_grad()
    def fuse(self):
        conv, conv1 = self.conv.fuse(), self.conv1.fuse()
        conv_w, conv_b = conv.weight, conv.bias
        conv1_w, conv1_b = conv1.weight, conv1.bias
        conv1_w = F.pad(conv1_w, [1, 1, 1, 1])
        identity = F.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device), [1, 1, 1, 1])
        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b
        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)
        return conv


'''RepViTBlock'''
class RepViTBlock(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs, skip_downsample=False):
        super(RepViTBlock, self).__init__()
        # assert
        assert stride in [1, 2]
        assert (hidden_dim == 2 * inp)
        # build modules
        self.identity = stride == 1 and inp == oup
        if stride == 2:
            if skip_downsample:
                stride = 1
            self.token_mixer = nn.Sequential(
                Conv2dBN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
                SqueezeExcitationConv2d(inp, 4, act_cfgs=[{'type': 'ReLU', 'inplace': True}, {'type': 'Sigmoid'}], makedivisible_args={'divisor': 1}) if use_se else nn.Identity(),
                Conv2dBN(inp, oup, ks=1, stride=1, pad=0)
            )
            self.channel_mixer = Residual(nn.Sequential(
                Conv2dBN(oup, 2 * oup, 1, 1, 0),
                nn.GELU() if use_hs else nn.GELU(),
                Conv2dBN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
            ))
        else:
            assert (self.identity)
            self.token_mixer = nn.Sequential(
                RepVGGDW(inp),
                SqueezeExcitationConv2d(inp, 4, act_cfgs=[{'type': 'ReLU', 'inplace': True}, {'type': 'Sigmoid'}], makedivisible_args={'divisor': 1}) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(nn.Sequential(
                Conv2dBN(inp, hidden_dim, 1, 1, 0),
                nn.GELU() if use_hs else nn.GELU(),
                Conv2dBN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
            ))
    '''forward'''
    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))


'''EdgeSAMRepViT'''
class EdgeSAMRepViT(nn.Module):
    arch_settings = {
        'm1': [
            [3, 2, 48, 1, 0, 1], [3, 2, 48, 0, 0, 1], [3, 2, 48, 0, 0, 1], [3, 2, 96, 0, 0, 2], [3, 2, 96, 1, 0, 1], [3, 2, 96, 0, 0, 1], [3, 2, 96, 0, 0, 1], 
            [3, 2, 192, 0, 1, 2], [3, 2, 192, 1, 1, 1], [3, 2, 192, 0, 1, 1], [3, 2, 192, 1, 1, 1], [3, 2, 192, 0, 1, 1], [3, 2, 192, 1, 1, 1], [3, 2, 192, 0, 1, 1], 
            [3, 2, 192, 1, 1, 1], [3, 2, 192, 0, 1, 1], [3, 2, 192, 1, 1, 1], [3, 2, 192, 0, 1, 1], [3, 2, 192, 1, 1, 1], [3, 2, 192, 0, 1, 1], [3, 2, 192, 1, 1, 1], 
            [3, 2, 192, 0, 1, 1], [3, 2, 192, 0, 1, 1], [3, 2, 384, 0, 1, 2], [3, 2, 384, 1, 1, 1], [3, 2, 384, 0, 1, 1]
        ],
        'm2': [
            [3, 2, 64, 1, 0, 1], [3, 2, 64, 0, 0, 1], [3, 2, 64, 0, 0, 1], [3, 2, 128, 0, 0, 2], [3, 2, 128, 1, 0, 1], [3, 2, 128, 0, 0, 1], [3, 2, 128, 0, 0, 1],
            [3, 2, 256, 0, 1, 2], [3, 2, 256, 1, 1, 1], [3, 2, 256, 0, 1, 1], [3, 2, 256, 1, 1, 1], [3, 2, 256, 0, 1, 1], [3, 2, 256, 1, 1, 1], [3, 2, 256, 0, 1, 1],
            [3, 2, 256, 1, 1, 1], [3, 2, 256, 0, 1, 1], [3, 2, 256, 1, 1, 1], [3, 2, 256, 0, 1, 1], [3, 2, 256, 1, 1, 1], [3, 2, 256, 0, 1, 1], [3, 2, 256, 0, 1, 1],
            [3, 2, 512, 0, 1, 2], [3, 2, 512, 1, 1, 1], [3, 2, 512, 0, 1, 1]
        ],
        'm3': [
            [3, 2, 64, 1, 0, 1], [3, 2, 64, 0, 0, 1], [3, 2, 64, 1, 0, 1], [3, 2, 64, 0, 0, 1], [3, 2, 64, 0, 0, 1], [3, 2, 128, 0, 0, 2], [3, 2, 128, 1, 0, 1],
            [3, 2, 128, 0, 0, 1], [3, 2, 128, 1, 0, 1], [3, 2, 128, 0, 0, 1], [3, 2, 128, 0, 0, 1], [3, 2, 256, 0, 1, 2], [3, 2, 256, 1, 1, 1], [3, 2, 256, 0, 1, 1],
            [3, 2, 256, 1, 1, 1], [3, 2, 256, 0, 1, 1], [3, 2, 256, 1, 1, 1], [3, 2, 256, 0, 1, 1], [3, 2, 256, 1, 1, 1], [3, 2, 256, 0, 1, 1], [3, 2, 256, 1, 1, 1],
            [3, 2, 256, 0, 1, 1], [3, 2, 256, 1, 1, 1], [3, 2, 256, 0, 1, 1], [3, 2, 256, 1, 1, 1], [3, 2, 256, 0, 1, 1], [3, 2, 256, 1, 1, 1], [3, 2, 256, 0, 1, 1],
            [3, 2, 256, 1, 1, 1], [3, 2, 256, 0, 1, 1], [3, 2, 256, 0, 1, 1], [3, 2, 512, 0, 1, 2], [3, 2, 512, 1, 1, 1], [3, 2, 512, 0, 1, 1]
        ],
    }
    def __init__(self, structure_type, arch, img_size=1024, upsample_mode='bicubic', pretrained=False, pretrained_model_path=''):
        super(RepViT, self).__init__()
        # set attributes
        self.arch = arch
        self.cfgs = self.arch_settings[arch]
        self.img_size = img_size
        self.structure_type = structure_type
        self.upsample_mode = upsample_mode
        self.pretrained = pretrained
        self.pretrained_model_path = pretrained_model_path
        # assert
        if structure_type in AUTO_ASSERT_STRUCTURE_TYPES:
            for key, value in AUTO_ASSERT_STRUCTURE_TYPES[structure_type].items():
                assert hasattr(self, key) and (getattr(self, key) == value)
        # building first layer
        input_channel = self.cfgs[0][2]
        patch_embed = nn.Sequential(
            Conv2dBN(3, input_channel // 2, 3, 2, 1), nn.GELU(), Conv2dBN(input_channel // 2, input_channel, 3, 2, 1)
        )
        layers = [patch_embed]
        # building inverted residual blocks
        block = RepViTBlock
        self.stage_idx = []
        prev_c = input_channel
        for idx, (k, t, c, use_se, use_hs, s) in enumerate(self.cfgs):
            output_channel = makedivisible(c, 8)
            exp_size = makedivisible(input_channel * t, 8)
            skip_downsample = False
            if c != prev_c:
                self.stage_idx.append(idx - 1)
                prev_c = c
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs, skip_downsample))
            input_channel = output_channel
        self.stage_idx.append(idx)
        self.features = nn.ModuleList(layers)
        # build fuse stages
        stage2_channels = makedivisible(self.cfgs[self.stage_idx[2]][2], 8)
        stage3_channels = makedivisible(self.cfgs[self.stage_idx[3]][2], 8)
        self.fuse_stage2 = nn.Conv2d(stage2_channels, 256, kernel_size=1, bias=False)
        self.fuse_stage3 = nn.Sequential(nn.Conv2d(stage3_channels, 256, kernel_size=1, bias=False))
        # build neck
        self.neck = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, bias=False), LayerNorm2d(256), nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False), LayerNorm2d(256),
        )
    '''forward'''
    def forward(self, x):
        counter = 0
        output_dict = dict()
        # patch_embed
        x = self.features[0](x)
        output_dict['stem'] = x
        # stages
        for idx, f in enumerate(self.features[1:]):
            x = f(x)
            if idx in self.stage_idx:
                output_dict[f'stage{counter}'] = x
                counter += 1
        # fuse
        x = self.fuse_stage2(output_dict['stage2']) + F.interpolate(self.fuse_stage3(output_dict['stage3']), scale_factor=2, mode=self.upsample_mode, align_corners=False) 
        # neck
        x = self.neck(x)
        # return
        return x