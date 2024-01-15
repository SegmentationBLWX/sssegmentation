'''
Function:
    Implementation of ERFNet
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...utils import loadpretrainedweights
from .bricks import BuildNormalization, BuildActivation


'''DEFAULT_MODEL_URLS'''
DEFAULT_MODEL_URLS = {}
'''AUTO_ASSERT_STRUCTURE_TYPES'''
AUTO_ASSERT_STRUCTURE_TYPES = {}


'''DownsamplerBlock'''
class DownsamplerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=None, act_cfg=None):
        super(DownsamplerBlock, self).__init__()
        self.norm_cfg, self.act_cfg = norm_cfg, act_cfg
        self.conv = nn.Conv2d(in_channels, out_channels - in_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg)
        self.act = BuildActivation(act_cfg)
    '''forward'''
    def forward(self, x):
        conv_out, pool_out = self.conv(x), self.pool(x)
        pool_out = F.interpolate(pool_out, size=conv_out.size()[2:], mode='bilinear', align_corners=False)
        output = torch.cat([conv_out, pool_out], dim=1)
        output = self.bn(output)
        output = self.act(output)
        return output


'''NonBottleneck1d'''
class NonBottleneck1d(nn.Module):
    def __init__(self, channels, drop_rate=0, dilation=1, num_conv_layer=2, norm_cfg=None, act_cfg=None):
        super(NonBottleneck1d, self).__init__()
        self.norm_cfg, self.act_cfg = norm_cfg, act_cfg
        self.act = BuildActivation(act_cfg)
        self.convs_layers = nn.ModuleList()
        for conv_layer in range(num_conv_layer):
            first_conv_padding = (1, 0) if conv_layer == 0 else (dilation, 0)
            first_conv_dilation = 1 if conv_layer == 0 else (dilation, 1)
            second_conv_padding = (0, 1) if conv_layer == 0 else (0, dilation)
            second_conv_dilation = 1 if conv_layer == 0 else (1, dilation)
            self.convs_layers.append(nn.Conv2d(channels, channels, kernel_size=(3, 1), stride=1, padding=first_conv_padding, bias=True, dilation=first_conv_dilation))
            self.convs_layers.append(self.act)
            self.convs_layers.append(nn.Conv2d(channels, channels, kernel_size=(1, 3), stride=1, padding=second_conv_padding, bias=True, dilation=second_conv_dilation))
            self.convs_layers.append(BuildNormalization(placeholder=channels, norm_cfg=norm_cfg))
            if conv_layer == 0: self.convs_layers.append(self.act)
            else: self.convs_layers.append(nn.Dropout(p=drop_rate))
    '''forward'''
    def forward(self, x):
        output = x
        for conv in self.convs_layers:
            output = conv(output)
        output = self.act(output + x)
        return output


'''UpsamplerBlock'''
class UpsamplerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=None, act_cfg=None):
        super(UpsamplerBlock, self).__init__()
        self.norm_cfg, self.act_cfg = norm_cfg, act_cfg
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg)
        self.act = BuildActivation(act_cfg)
    '''forward'''
    def forward(self, x):
        output = self.conv(x)
        output = self.bn(output)
        output = self.act(output)
        return output


'''ERFNet'''
class ERFNet(nn.Module):
    def __init__(self, structure_type, in_channels=3, enc_downsample_channels=(16, 64, 128), enc_stage_non_bottlenecks=(5, 8), enc_non_bottleneck_dilations=(2, 4, 8, 16),
                 enc_non_bottleneck_channels=(64, 128), dec_upsample_channels=(64, 16), dec_stages_non_bottleneck=(2, 2), dec_non_bottleneck_channels=(64, 16),
                 dropout_ratio=0.1, norm_cfg={'type': 'SyncBatchNorm'}, act_cfg={'type': 'PReLU'}, pretrained=False, pretrained_model_path=''):
        super(ERFNet, self).__init__()
        # set attributes
        self.structure_type = structure_type
        self.in_channels = in_channels
        self.enc_downsample_channels = enc_downsample_channels
        self.enc_stage_non_bottlenecks = enc_stage_non_bottlenecks
        self.enc_non_bottleneck_dilations = enc_non_bottleneck_dilations
        self.enc_non_bottleneck_channels = enc_non_bottleneck_channels
        self.dec_upsample_channels = dec_upsample_channels
        self.dec_stages_non_bottleneck = dec_stages_non_bottleneck
        self.dec_non_bottleneck_channels = dec_non_bottleneck_channels
        self.dropout_ratio = dropout_ratio
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.pretrained = pretrained
        self.pretrained_model_path = pretrained_model_path
        # assert
        assert len(enc_downsample_channels) == len(dec_upsample_channels) + 1
        assert len(enc_downsample_channels) == len(enc_stage_non_bottlenecks) + 1
        assert len(enc_downsample_channels) == len(enc_non_bottleneck_channels) + 1
        assert enc_stage_non_bottlenecks[-1] % len(enc_non_bottleneck_dilations) == 0
        assert len(dec_upsample_channels) == len(dec_stages_non_bottleneck)
        assert len(dec_stages_non_bottleneck) == len(dec_non_bottleneck_channels)
        if structure_type in AUTO_ASSERT_STRUCTURE_TYPES:
            for key, value in AUTO_ASSERT_STRUCTURE_TYPES[structure_type].items():
                assert hasattr(self, key) and (getattr(self, key) == value)
        # set encoder and decoder
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        # --encoder
        self.encoder.append(DownsamplerBlock(in_channels, enc_downsample_channels[0], norm_cfg=norm_cfg, act_cfg=act_cfg))
        for i in range(len(enc_downsample_channels) - 1):
            self.encoder.append(DownsamplerBlock(enc_downsample_channels[i], enc_downsample_channels[i + 1], norm_cfg=norm_cfg, act_cfg=act_cfg))
            if i == len(enc_downsample_channels) - 2:
                iteration_times = int(enc_stage_non_bottlenecks[-1] / len(enc_non_bottleneck_dilations))
                for j in range(iteration_times):
                    for k in range(len(enc_non_bottleneck_dilations)):
                        self.encoder.append(NonBottleneck1d(enc_downsample_channels[-1], dropout_ratio, enc_non_bottleneck_dilations[k], norm_cfg=norm_cfg, act_cfg=act_cfg))
            else:
                for j in range(enc_stage_non_bottlenecks[i]):
                    self.encoder.append(NonBottleneck1d(enc_downsample_channels[i + 1], dropout_ratio, norm_cfg=norm_cfg, act_cfg=act_cfg))
        # --decoder
        for i in range(len(dec_upsample_channels)):
            if i == 0: self.decoder.append(UpsamplerBlock(enc_downsample_channels[-1], dec_non_bottleneck_channels[i], norm_cfg=norm_cfg, act_cfg=act_cfg))
            else: self.decoder.append(UpsamplerBlock(dec_non_bottleneck_channels[i - 1], dec_non_bottleneck_channels[i], norm_cfg=norm_cfg, act_cfg=act_cfg))
            for j in range(dec_stages_non_bottleneck[i]): self.decoder.append(NonBottleneck1d(dec_non_bottleneck_channels[i], norm_cfg=norm_cfg, act_cfg=act_cfg))
        # load pretrained weights
        if pretrained:
            state_dict = loadpretrainedweights(
                structure_type=structure_type, pretrained_model_path=pretrained_model_path, default_model_urls=DEFAULT_MODEL_URLS
            )
            self.load_state_dict(state_dict, strict=False)
    '''forward'''
    def forward(self, x):
        for enc in self.encoder: x = enc(x)
        for dec in self.decoder: x = dec(x)
        return [x]