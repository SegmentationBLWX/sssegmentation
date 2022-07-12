'''
Function:
    Implementation of UNet
Author:
    Zhenchao Jin
'''
import os
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from .bricks import BuildNormalization, BuildActivation, constructnormcfg


'''model urls'''
model_urls = {}


'''Basic convolutional block for UNet'''
class BasicConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs=2, stride=1, dilation=1, norm_cfg=None, act_cfg=None):
        super(BasicConvBlock, self).__init__()
        convs = []
        for i in range(num_convs):
            in_c, out_c = in_channels if i == 0 else out_channels, out_channels
            s, d, p = stride if i == 0 else 1, 1 if i == 0 else dilation, 1 if i == 0 else dilation
            conv = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=s, padding=p, dilation=d, bias=False),
                BuildNormalization(constructnormcfg(placeholder=out_c, norm_cfg=norm_cfg)),
                BuildActivation(act_cfg),
            )
            convs.append(conv)
        self.convs = nn.Sequential(*convs)
    '''forward'''
    def forward(self, x):
        out = self.convs(x)
        return out


'''Deconvolution upsample module in decoder for UNet (2X upsample)'''
class DeconvModule(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=None, act_cfg=None, kernel_size=4, scale_factor=2):
        super(DeconvModule, self).__init__()
        assert (kernel_size - scale_factor >= 0) and (kernel_size - scale_factor) % 2 == 0
        self.deconv_upsamping = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=scale_factor, padding=(kernel_size - scale_factor) // 2),
            BuildNormalization(constructnormcfg(placeholder=out_channels, norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
        )
    '''forward'''
    def forward(self, x):
        out = self.deconv_upsamping(x)
        return out


'''Interpolation upsample module in decoder for UNet'''
class InterpConv(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=None, act_cfg=None, conv_first=False, kernel_size=1, stride=1, padding=0, 
                 upsample_cfg=dict(scale_factor=2, mode='bilinear', align_corners=False)):
        super(InterpConv, self).__init__()
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            BuildNormalization(constructnormcfg(placeholder=out_channels, norm_cfg=norm_cfg)),
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


'''Upsample convolution block in decoder for UNet'''
class UpConvBlock(nn.Module):
    def __init__(self, conv_block, in_channels, skip_channels, out_channels, num_convs=2, stride=1, dilation=1,
                 norm_cfg=None, act_cfg=None, upsample_type='InterpConv'):
        super(UpConvBlock, self).__init__()
        supported_upsamples = {
            'InterpConv': InterpConv,
            'DeconvModule': DeconvModule,
        }
        self.conv_block = conv_block(
            in_channels=2 * skip_channels,
            out_channels=out_channels,
            num_convs=num_convs,
            stride=stride,
            dilation=dilation,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        if upsample_type is not None:
            assert upsample_type in supported_upsamples, 'unsupport upsample_type %s' % upsample_type
            self.upsample = supported_upsamples[upsample_type](
                in_channels=in_channels,
                out_channels=skip_channels,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
        else:
            self.upsample = nn.Sequential(
                nn.Conv2d(in_channels, skip_channels, kernel_size=1, stride=1, padding=0, bias=False),
                BuildNormalization(constructnormcfg(placeholder=skip_channels, norm_cfg=norm_cfg)),
                BuildActivation(act_cfg),
            )
    '''forward'''
    def forward(self, skip, x):
        x = self.upsample(x)
        out = torch.cat([skip, x], dim=1)
        out = self.conv_block(out)
        return out


'''UNet backbone'''
class UNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, num_stages=5, strides=(1, 1, 1, 1, 1), enc_num_convs=(2, 2, 2, 2, 2), dec_num_convs=(2, 2, 2, 2),
                 downsamples=(True, True, True, True), enc_dilations=(1, 1, 1, 1, 1), dec_dilations=(1, 1, 1, 1), norm_cfg=None, act_cfg=None, upsample_type='InterpConv'):
        super(UNet, self).__init__()
        assert (len(strides) == num_stages) and (len(enc_num_convs) == num_stages) \
            and (len(dec_num_convs) == (num_stages - 1)) and (len(downsamples) == (num_stages - 1)) \
            and (len(enc_dilations) == num_stages) and len(dec_dilations) == (num_stages - 1)
        # set attrs
        self.num_stages = num_stages
        self.strides = strides
        self.downsamples = downsamples
        self.base_channels = base_channels
        # build encoder and decoder
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(num_stages):
            enc_conv_block = []
            if i != 0:
                if strides[i] == 1 and downsamples[i - 1]:
                    enc_conv_block.append(nn.MaxPool2d(kernel_size=2))
                upsample = (strides[i] != 1 or downsamples[i - 1])
                self.decoder.append(
                    UpConvBlock(
                        conv_block=BasicConvBlock,
                        in_channels=base_channels * 2**i,
                        skip_channels=base_channels * 2**(i - 1),
                        out_channels=base_channels * 2**(i - 1),
                        num_convs=dec_num_convs[i - 1],
                        stride=1,
                        dilation=dec_dilations[i - 1],
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        upsample_type=upsample_type if upsample else None,
                    )
                )
            enc_conv_block.append(
                BasicConvBlock(
                    in_channels=in_channels,
                    out_channels=base_channels * 2**i,
                    num_convs=enc_num_convs[i],
                    stride=strides[i],
                    dilation=enc_dilations[i],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
            )
            self.encoder.append((nn.Sequential(*enc_conv_block)))
            in_channels = base_channels * 2**i
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


'''BuildUNet'''
def BuildUNet(unet_cfg):
    # assert whether support
    unet_type = unet_cfg.pop('type')
    # parse cfg
    default_cfg = {
        'in_channels': 3,
        'base_channels': 64,
        'num_stages': 5,
        'strides': (1, 1, 1, 1, 1),
        'enc_num_convs': (2, 2, 2, 2, 2),
        'dec_num_convs': (2, 2, 2, 2),
        'downsamples': (True, True, True, True),
        'enc_dilations': (1, 1, 1, 1, 1),
        'dec_dilations': (1, 1, 1, 1),
        'norm_cfg': None,
        'act_cfg': {'type': 'relu', 'inplace': True},
        'upsample_type': 'InterpConv',
        'pretrained': False,
        'pretrained_model_path': '',
    }
    for key, value in unet_cfg.items():
        if key in default_cfg: 
            default_cfg.update({key: value})
    # obtain unet_cfg
    unet_cfg = default_cfg.copy()
    pretrained = unet_cfg.pop('pretrained')
    pretrained_model_path = unet_cfg.pop('pretrained_model_path')
    # obtain the instanced unet
    model = UNet(**unet_cfg)
    # load weights of pretrained model
    if pretrained and os.path.exists(pretrained_model_path):
        checkpoint = torch.load(pretrained_model_path)
        if 'state_dict' in checkpoint: 
            state_dict = checkpoint['state_dict']
        else: 
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    elif pretrained:
        checkpoint = model_zoo.load_url(model_urls[unet_type])
        if 'state_dict' in checkpoint: 
            state_dict = checkpoint['state_dict']
        else: 
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    # return the model
    return model