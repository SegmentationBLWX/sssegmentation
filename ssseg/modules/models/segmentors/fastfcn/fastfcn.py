'''
Function:
    Implementation of FastFCN
Author:
    Zhenchao Jin
'''
import torch.nn as nn
from .jpu import JPU
from ..fcn import FCN
from ..encnet import ENCNet
from ..pspnet import PSPNet
from ..deeplabv3 import Deeplabv3


'''FastFCN'''
class FastFCN(nn.Module):
    def __init__(self, cfg, mode):
        super(FastFCN, self).__init__()
        self.align_corners, self.norm_cfg, self.act_cfg, head_cfg = cfg['align_corners'], cfg['norm_cfg'], cfg['act_cfg'], cfg['head']
        # build segmentor
        supported_models = {
            'fcn': FCN,
            'encnet': ENCNet,
            'pspnet': PSPNet,
            'deeplabv3': Deeplabv3,
        }
        model_type = cfg['segmentor']
        assert model_type in supported_models, 'unsupport model_type %s' % model_type
        self.segmentor = supported_models[model_type](cfg, mode)
        # build jpu neck
        jpu_cfg = head_cfg['jpu']
        if 'act_cfg' not in jpu_cfg: jpu_cfg.update({'act_cfg': self.act_cfg})
        if 'norm_cfg' not in jpu_cfg: jpu_cfg.update({'norm_cfg': self.norm_cfg})
        if 'align_corners' not in jpu_cfg: jpu_cfg.update({'align_corners': self.align_corners})
        self.jpu_neck = JPU(**jpu_cfg)
        self.segmentor.transforminputs = self.transforminputs
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
    '''forward'''
    def forward(self, x, targets=None, losses_cfg=None, **kwargs):
        return self.segmentor(x, targets, losses_cfg, **kwargs)
    '''transform inputs'''
    def transforminputs(self, x_list, selected_indices=None):
        if selected_indices is None:
            if self.cfg['backbone']['series'] in ['hrnet']:
                selected_indices = (0, 0, 0, 0)
            else:
                selected_indices = (0, 1, 2, 3)
        outs = []
        for idx in selected_indices:
            outs.append(x_list[idx])
        outs = self.jpu_neck(outs)
        return outs
    '''return all layers with learnable parameters'''
    def alllayers(self):
        require_training_layers = self.segmentor.alllayers()
        require_training_layers['jpu_neck'] = self.jpu_neck
        return require_training_layers