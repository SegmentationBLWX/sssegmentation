'''
Function:
    Implementation of Auxiliary Decoder
Author:
    Zhenchao Jin
'''
import copy
import torch.nn as nn
from ....utils import BaseModuleBuilder
from ...backbones import BuildNormalization, BuildActivation


'''DefaultAuxiliaryDecoder'''
class DefaultAuxiliaryDecoder(nn.Sequential):
    def __init__(self, **kwargs):
        auxiliary_cfg = kwargs
        num_convs, dec = auxiliary_cfg.get('num_convs', 1), []
        for idx in range(num_convs):
            if idx == 0:
                dec.append(nn.Conv2d(auxiliary_cfg['in_channels'], auxiliary_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False))
            else:
                dec.append(nn.Conv2d(auxiliary_cfg['out_channels'], auxiliary_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False))
            dec.append(BuildNormalization(placeholder=auxiliary_cfg['out_channels'], norm_cfg=auxiliary_cfg['norm_cfg']))
            dec.append(BuildActivation(auxiliary_cfg['act_cfg']))
            if 'upsample' in auxiliary_cfg:
                dec.append(nn.Upsample(**auxiliary_cfg['upsample']))
        dec.append(nn.Dropout2d(auxiliary_cfg['dropout']))
        if num_convs > 0:
            dec.append(nn.Conv2d(auxiliary_cfg['out_channels'], auxiliary_cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        else:
            dec.append(nn.Conv2d(auxiliary_cfg['in_channels'], auxiliary_cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        super(DefaultAuxiliaryDecoder, self).__init__(*dec)


'''AuxiliaryDecoderBuilder'''
class AuxiliaryDecoderBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'DefaultAuxiliaryDecoder': DefaultAuxiliaryDecoder, 
    }
    '''build'''
    def build(self, auxiliary_cfg, norm_cfg, act_cfg, num_classes):
        auxiliary_cfg = copy.deepcopy(auxiliary_cfg)
        if 'type' not in auxiliary_cfg:
            auxiliary_cfg['type'] = 'DefaultAuxiliaryDecoder'
        if 'norm_cfg' not in auxiliary_cfg:
            auxiliary_cfg['norm_cfg'] = norm_cfg
        if 'act_cfg' not in auxiliary_cfg:
            auxiliary_cfg['act_cfg'] = act_cfg
        if 'num_classes' not in auxiliary_cfg:
            auxiliary_cfg['num_classes'] = num_classes
        return super().build(auxiliary_cfg)


'''BuildAuxiliaryDecoder'''
BuildAuxiliaryDecoder = AuxiliaryDecoderBuilder().build