'''
Function:
    Implementation of ENCNet
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import BaseSegmentor
from .contextencoding import ContextEncoding
from ...backbones import BuildActivation, BuildNormalization, constructnormcfg


'''ENCNet'''
class ENCNet(BaseSegmentor):
    def __init__(self, cfg, mode):
        super(ENCNet, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        # build encoding
        # --base structurs
        self.bottleneck = nn.Sequential(
            nn.Conv2d(head_cfg['in_channels_list'][-1], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(constructnormcfg(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
        )
        self.enc_module = ContextEncoding(
            in_channels=head_cfg['feats_channels'],
            num_codes=head_cfg['num_codes'],
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        # --extra structures
        extra_cfg = head_cfg['extra']
        if extra_cfg['add_lateral']:
            self.lateral_convs = nn.ModuleList()
            for in_channels in head_cfg['in_channels_list'][:-1]:
                self.lateral_convs.append(
                    nn.Conv2d(in_channels, head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0),
                    BuildNormalization(constructnormcfg(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg)),
                    BuildActivation(act_cfg),
                )
            self.fusion = nn.Sequential(
                nn.Conv2d(len(head_cfg['in_channels_list']) * head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1),
                BuildNormalization(constructnormcfg(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg)),
                BuildActivation(act_cfg),
            )
        if extra_cfg['use_se_loss']:
            self.se_layer = nn.Linear(head_cfg['feats_channels'], cfg['num_classes'])
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
        self.layer_names = ['backbone_net', 'bottleneck', 'enc_module', 'decoder', 'lateral_convs', 'fusion', 'se_layer', 'auxiliary_decoder']
    '''forward'''
    def forward(self, x, targets=None, losses_cfg=None):
        img_size = x.size(2), x.size(3)
        # feed to backbone network
        backbone_outputs = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to context encoding
        feats = self.bottleneck(backbone_outputs[-1])
        if hasattr(self, 'lateral_convs'):
            lateral_outs = [
                F.interpolate(lateral_conv(backbone_outputs[idx]), size=feats.shape[2:], mode='bilinear', align_corners=self.align_corners) for idx, lateral_conv in enumerate(self.lateral_convs)
            ]
            feats = self.fusion(torch.cat([feats, *lateral_outs], dim=1))
        encode_feats, feats = self.enc_module(feats)
        if hasattr(self, 'se_layer'):
            predictions_se = self.se_layer(encode_feats)
        # feed to decoder
        predictions = self.decoder(feats)
        # forward according to the mode
        if self.mode == 'TRAIN':
            outputs_dict = self.forwardtrain(
                predictions=predictions,
                targets=targets,
                backbone_outputs=backbone_outputs,
                losses_cfg=losses_cfg,
                img_size=img_size,
                compute_loss=False,
            )
            if hasattr(self, 'se_layer'):
                outputs_dict.update({'loss_se': predictions_se})
            return self.calculatelosses(
                predictions=outputs_dict, 
                targets=targets, 
                losses_cfg=losses_cfg
            )
        return predictions
    '''convert to onehot labels'''
    def onehot(self, labels, num_classes):
        batch_size = labels.size(0)
        labels_onehot = labels.new_zeros((batch_size, num_classes))
        for i in range(batch_size):
            hist = labels[i].float().histc(bins=num_classes, min=0, max=num_classes-1)
            labels_onehot[i] = hist > 0
        return labels_onehot