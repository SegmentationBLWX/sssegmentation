'''
Function:
    Implementation of FastFCN
Author:
    Zhenchao Jin
'''
from .jpu import JPU
from ..fcn import FCN
from ..encnet import ENCNet
from ..pspnet import PSPNet
from ..base import BaseSegmentor
from ..deeplabv3 import Deeplabv3


'''FastFCN'''
class FastFCN(BaseSegmentor):
    def __init__(self, cfg, mode):
        backbone = cfg.pop('backbone')
        super(FastFCN, self).__init__(cfg=cfg, mode=mode)
        cfg['backbone'] = backbone
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        # build segmentor
        supported_segmentors = {
            'FCN': FCN, 'ENCNet': ENCNet, 'PSPNet': PSPNet, 'Deeplabv3': Deeplabv3,
        }
        self.segmentor = supported_segmentors[cfg['segmentor']](cfg, mode)
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
    def forward(self, data_meta, **kwargs):
        return self.segmentor(data_meta, **kwargs)
    '''transforminputs'''
    def transforminputs(self, x_list, selected_indices=None):
        if selected_indices is None:
            if self.cfg['backbone']['type'] in ['HRNet']:
                selected_indices = (0, 0, 0, 0)
            else:
                selected_indices = (0, 1, 2, 3)
        outs = []
        for idx in selected_indices:
            outs.append(x_list[idx])
        outs = self.jpu_neck(outs)
        return outs