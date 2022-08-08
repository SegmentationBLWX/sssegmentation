'''
Function:
    Implementation of Deeplabv3
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
from .aspp import ASPP
from ..base import BaseSegmentor


'''Deeplabv3'''
class Deeplabv3(BaseSegmentor):
    def __init__(self, cfg, mode):
        super(Deeplabv3, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        # build aspp net
        aspp_cfg = {
            'in_channels': head_cfg['in_channels'],
            'out_channels': head_cfg['feats_channels'],
            'dilations': head_cfg['dilations'],
            'align_corners': align_corners,
            'norm_cfg': copy.deepcopy(norm_cfg),
            'act_cfg': copy.deepcopy(act_cfg),
        }
        self.aspp_net = ASPP(**aspp_cfg)
        # build decoder
        self.decoder = nn.Sequential(
            nn.Dropout2d(head_cfg['dropout']),
            nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        # build auxiliary decoder
        self.setauxiliarydecoder(cfg['auxiliary'])
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
        # layer names for training tricks
        self.layer_names = ['backbone_net', 'aspp_net', 'decoder', 'auxiliary_decoder']
    '''forward'''
    def forward(self, x, targets=None, losses_cfg=None):
        img_size = x.size(2), x.size(3)
        # feed to backbone network
        backbone_outputs = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to aspp
        aspp_out = self.aspp_net(backbone_outputs[-1])
        # feed to decoder
        predictions = self.decoder(aspp_out)
        # forward according to the mode
        if self.mode == 'TRAIN':
            loss, losses_log_dict = self.forwardtrain(
                predictions=predictions,
                targets=targets,
                backbone_outputs=backbone_outputs,
                losses_cfg=losses_cfg,
                img_size=img_size,
            )
            return loss, losses_log_dict
        return predictions