'''
Function:
    Implementation of UNet
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


'''BasicConvBlock'''
class BasicConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs=2, stride=1, dilation=1, norm_cfg=None, act_cfg=None):
        super(BasicConvBlock, self).__init__()
        convs = []
        for i in range(num_convs):
            in_c, out_c = in_channels if i == 0 else out_channels, out_channels
            s, d, p = stride if i == 0 else 1, 1 if i == 0 else dilation, 1 if i == 0 else dilation
            conv = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=s, padding=p, dilation=d, bias=False),
                BuildNormalization(placeholder=out_c, norm_cfg=norm_cfg),
                BuildActivation(act_cfg),
            )
            convs.append(conv)
        self.convs = nn.Sequential(*convs)
    '''forward'''
    def forward(self, x):
        out = self.convs(x)
        return out


'''DeconvModule'''
class DeconvModule(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=None, act_cfg=None, kernel_size=4, scale_factor=2):
        super(DeconvModule, self).__init__()
        assert (kernel_size - scale_factor >= 0) and (kernel_size - scale_factor) % 2 == 0
        self.deconv_upsamping = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=scale_factor, padding=(kernel_size - scale_factor) // 2),
            BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
    '''forward'''
    def forward(self, x):
        out = self.deconv_upsamping(x)
        return out


'''InterpConv'''
class InterpConv(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=None, act_cfg=None, conv_first=False, kernel_size=1, stride=1, padding=0, 
                 upsample_cfg=dict(scale_factor=2, mode='bilinear', align_corners=False)):
        super(InterpConv, self).__init__()
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        upsample = nn.Upsample(**upsample_cfg)
        if conv_first:
            self.interp_upsample = nn.Sequential(conv, upsample)
        else:
            self.interp_upsample = nn.Sequential(upsample, conv)
    '''forward'''
    def forward(self, x):
        out = self.interp_upsample(x)
        return out


'''UpConvBlock'''
class UpConvBlock(nn.Module):
    def __init__(self, conv_block, in_channels, skip_channels, out_channels, num_convs=2, stride=1, dilation=1,
                 norm_cfg=None, act_cfg=None, upsample_type='InterpConv'):
        super(UpConvBlock, self).__init__()
        supported_upsamples = {
            'InterpConv': InterpConv, 'DeconvModule': DeconvModule,
        }
        self.conv_block = conv_block(
            in_channels=2 * skip_channels, out_channels=out_channels, num_convs=num_convs, stride=stride,
            dilation=dilation, norm_cfg=norm_cfg, act_cfg=act_cfg
        )
        if upsample_type is not None:
            assert upsample_type in supported_upsamples, 'unsupport upsample_type %s' % upsample_type
            self.upsample = supported_upsamples[upsample_type](in_channels=in_channels, out_channels=skip_channels, norm_cfg=norm_cfg, act_cfg=act_cfg)
        else:
            self.upsample = nn.Sequential(
                nn.Conv2d(in_channels, skip_channels, kernel_size=1, stride=1, padding=0, bias=False),
                BuildNormalization(placeholder=skip_channels, norm_cfg=norm_cfg),
                BuildActivation(act_cfg),
            )
    '''forward'''
    def forward(self, skip, x):
        x = self.upsample(x)
        out = torch.cat([skip, x], dim=1)
        out = self.conv_block(out)
        return out


'''UNet'''
class UNet(nn.Module):
    def __init__(self, structure_type, in_channels=3, base_channels=64, num_stages=5, strides=(1, 1, 1, 1, 1), enc_num_convs=(2, 2, 2, 2, 2), dec_num_convs=(2, 2, 2, 2),
                 downsamples=(True, True, True, True), enc_dilations=(1, 1, 1, 1, 1), dec_dilations=(1, 1, 1, 1), norm_cfg={'type': 'SyncBatchNorm'}, 
                 act_cfg={'type': 'ReLU', 'inplace': True}, upsample_type='InterpConv', pretrained=False, pretrained_model_path=''):
        super(UNet, self).__init__()
        # set attributes
        self.structure_type = structure_type
        self.in_channels = in_channels
        self.enc_num_convs = enc_num_convs
        self.dec_num_convs = dec_num_convs
        self.num_stages = num_stages
        self.strides = strides
        self.downsamples = downsamples
        self.base_channels = base_channels
        self.enc_dilations = enc_dilations
        self.dec_dilations = dec_dilations
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.upsample_type = upsample_type
        self.pretrained = pretrained
        self.pretrained_model_path = pretrained_model_path
        # assert
        assert (len(strides) == num_stages) and (len(enc_num_convs) == num_stages) \
            and (len(dec_num_convs) == (num_stages - 1)) and (len(downsamples) == (num_stages - 1)) \
            and (len(enc_dilations) == num_stages) and len(dec_dilations) == (num_stages - 1)
        if structure_type in AUTO_ASSERT_STRUCTURE_TYPES:
            for key, value in AUTO_ASSERT_STRUCTURE_TYPES[structure_type].items():
                assert hasattr(self, key) and (getattr(self, key) == value)
        # build encoder and decoder
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(num_stages):
            enc_conv_block = []
            if i != 0:
                if strides[i] == 1 and downsamples[i - 1]:
                    enc_conv_block.append(nn.MaxPool2d(kernel_size=2))
                upsample = (strides[i] != 1 or downsamples[i - 1])
                self.decoder.append(UpConvBlock(
                    conv_block=BasicConvBlock, in_channels=base_channels * 2**i, skip_channels=base_channels * 2**(i - 1), 
                    out_channels=base_channels * 2**(i - 1), num_convs=dec_num_convs[i - 1], stride=1, dilation=dec_dilations[i - 1], 
                    norm_cfg=norm_cfg, act_cfg=act_cfg, upsample_type=upsample_type if upsample else None,
                ))
            enc_conv_block.append(BasicConvBlock(
                in_channels=in_channels, out_channels=base_channels * 2**i, num_convs=enc_num_convs[i], stride=strides[i],
                dilation=enc_dilations[i], norm_cfg=norm_cfg, act_cfg=act_cfg,
            ))
            self.encoder.append((nn.Sequential(*enc_conv_block)))
            in_channels = base_channels * 2**i
        # load pretrained weights
        if pretrained:
            state_dict = loadpretrainedweights(
                structure_type=structure_type, pretrained_model_path=pretrained_model_path, default_model_urls=DEFAULT_MODEL_URLS
            )
            self.load_state_dict(state_dict, strict=False)
    '''forward'''
    def forward(self, x):
        self.checkinputdivisible(x)
        enc_outs = []
        for enc in self.encoder:
            x = enc(x)
            enc_outs.append(x)
        dec_outs = [x]
        for i in reversed(range(len(self.decoder))):
            x = self.decoder[i](enc_outs[i], x)
            dec_outs.append(x)
        return dec_outs
    '''check input divisible'''
    def checkinputdivisible(self, x):
        h, w = x.shape[-2:]
        whole_downsample_rate = 1
        for i in range(1, self.num_stages):
            if self.strides[i] == 2 or self.downsamples[i - 1]:
                whole_downsample_rate *= 2
        assert (h % whole_downsample_rate == 0) and (w % whole_downsample_rate == 0)