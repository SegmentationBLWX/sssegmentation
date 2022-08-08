'''
Function:
    Implementation of ICNet
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
from .icneck import ICNeck
from ..base import BaseSegmentor
from .icnetencoder import ICNetEncoder
from ...backbones import BuildActivation, BuildNormalization, constructnormcfg


'''ICNet'''
class ICNet(BaseSegmentor):
    def __init__(self, cfg, mode):
        super(ICNet, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        # build encoder
        delattr(self, 'backbone_net')
        encoder_cfg = head_cfg['encoder']
        encoder_cfg.update({'backbone_cfg': cfg['backbone']})
        if 'act_cfg' not in encoder_cfg: encoder_cfg.update({'act_cfg': act_cfg})
        if 'norm_cfg' not in encoder_cfg: encoder_cfg.update({'norm_cfg': norm_cfg})
        if 'align_corners' not in encoder_cfg: encoder_cfg.update({'align_corners': align_corners})
        self.backbone_net = ICNetEncoder(**encoder_cfg)
        # build neck
        neck_cfg = {
            'in_channels_list': head_cfg['in_channels_list'],
            'out_channels': head_cfg['feats_channels'],
            'act_cfg': act_cfg.copy(),
            'norm_cfg': norm_cfg.copy(),
            'align_corners': align_corners,
        }
        self.neck = ICNeck(**neck_cfg)
        # build decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(constructnormcfg(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
            nn.Dropout2d(head_cfg['dropout']),
            nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        # build auxiliary decoder
        self.setauxiliarydecoder(cfg['auxiliary'])
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
        # layer names for training tricks
        self.layer_names = ['backbone_net', 'neck', 'decoder', 'auxiliary_decoder']
    '''forward'''
    def forward(self, x, targets=None, losses_cfg=None):
        img_size = x.size(2), x.size(3)
        # feed to backbone network
        backbone_outputs = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to neck
        backbone_outputs = self.neck(backbone_outputs)
        # feed to decoder
        predictions = self.decoder(backbone_outputs[-1])
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