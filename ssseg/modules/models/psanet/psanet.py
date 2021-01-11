'''
Function:
    Implementation of PSANet
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...backbones import *
from ..base import BaseModel
from mmcv.ops import PSAMask


'''PSANet'''
class PSANet(BaseModel):
    def __init__(self, cfg, **kwargs):
        super(PSANet, self).__init__(cfg, **kwargs)
        align_corners, norm_cfg, act_cfg = self.align_corners, self.norm_cfg, self.act_cfg
        # build psa
        psa_cfg = cfg['psa']
        assert psa_cfg['type'] in ['collect', 'distribute', 'bi-direction']
        mask_h, mask_w = psa_cfg['mask_size']
        if 'normalization_factor' not in psa_cfg:
            psa_cfg['normalization_factor'] = mask_h * mask_w
        self.reduce = nn.Sequential(
            nn.Conv2d(psa_cfg['in_channels'], psa_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(norm_cfg['type'], (psa_cfg['out_channels'], norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts']),
        )
        self.attention = nn.Sequential(
            nn.Conv2d(psa_cfg['out_channels'], psa_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(norm_cfg['type'], (psa_cfg['out_channels'], norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts']),
            nn.Conv2d(psa_cfg['out_channels'], mask_h * mask_w, kernel_size=1, stride=1, padding=0, bias=False),
        )
        if psa_cfg['type'] == 'bi-direction':
            self.reduce_p = nn.Sequential(
                nn.Conv2d(psa_cfg['in_channels'], psa_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False),
                BuildNormalization(norm_cfg['type'], (psa_cfg['out_channels'], norm_cfg['opts'])),
                BuildActivation(act_cfg['type'], **act_cfg['opts']),
            )
            self.attention_p = nn.Sequential(
                nn.Conv2d(psa_cfg['out_channels'], psa_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False),
                BuildNormalization(norm_cfg['type'], (psa_cfg['out_channels'], norm_cfg['opts'])),
                BuildActivation(act_cfg['type'], **act_cfg['opts']),
                nn.Conv2d(psa_cfg['out_channels'], mask_h * mask_w, kernel_size=1, stride=1, padding=0, bias=False),
            )
            if not psa_cfg['compact']:
                self.psamask_collect = PSAMask('collect', psa_cfg['mask_size'])
                self.psamask_distribute = PSAMask('distribute', psa_cfg['mask_size'])
        else:
            if not psa_cfg['compact']:
                self.psamask = PSAMask(psa_cfg['type'], psa_cfg['mask_size'])
        self.proj = nn.Sequential(
            nn.Conv2d(psa_cfg['out_channels'] * (2 if psa_cfg['type'] == 'bi-direction' else 1), psa_cfg['in_channels'], kernel_size=1, stride=1, padding=1, bias=False),
            BuildNormalization(norm_cfg['type'], (psa_cfg['in_channels'], norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts']),
        )
        # build decoder
        decoder_cfg = cfg['decoder']
        self.decoder = nn.Sequential(
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(norm_cfg['type'], (decoder_cfg['out_channels'], norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts']),
            nn.Dropout2d(decoder_cfg['dropout']),
            nn.Conv2d(decoder_cfg['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        # build auxiliary decoder
        auxiliary_cfg = cfg['auxiliary']
        self.auxiliary_decoder = nn.Sequential(
            nn.Conv2d(auxiliary_cfg['in_channels'], auxiliary_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(norm_cfg['type'], (auxiliary_cfg['out_channels'], norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts']),
            nn.Dropout2d(auxiliary_cfg['dropout']),
            nn.Conv2d(auxiliary_cfg['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
    '''forward'''
    def forward(self, x, targets=None, losses_cfg=None):
        h_input, w_input = x.size(2), x.size(3)
        # feed to backbone network
        x3, x4 = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to psa
        identity = x4
        shrink_factor, align_corners = self.cfg['psa']['shrink_factor'], self.align_corners
        if self.cfg['psa']['type'] in ['collect', 'distribute']:
            out = self.reduce(x4)
            n, c, h, w = out.size()
            if shrink_factor != 1:
                if h % shrink_factor and w % shrink_factor:
                    h = (h - 1) // shrink_factor + 1
                    w = (w - 1) // shrink_factor + 1
                    align_corners = True
                else:
                    h = h // shrink_factor
                    w = w // shrink_factor
                    align_corners = False
                out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=align_corners)
            y = self.attention(out)
            if self.cfg['psa']['compact']:
                if self.cfg['psa']['type'] == 'collect':
                    y = y.view(n, h * w, h * w).transpose(1, 2).view(n, h * w, h, w)
            else:
                y = self.psamask(y)
            if self.cfg['psa']['psa_softmax']:
                y = F.softmax(y, dim=1)
            out = torch.bmm(out.view(n, c, h * w), y.view(n, h * w, h * w)).view(n, c, h, w) * (1.0 / self.cfg['psa']['normalization_factor'])
        else:
            x_col = self.reduce(x4)
            x_dis = self.reduce_p(x4)
            n, c, h, w = x_col.size()
            if shrink_factor != 1:
                if h % shrink_factor and w % shrink_factor:
                    h = (h - 1) // shrink_factor + 1
                    w = (w - 1) // shrink_factor + 1
                    align_corners = True
                else:
                    h = h // shrink_factor
                    w = w // shrink_factor
                    align_corners = False
                x_col = F.interpolate(x_col, size=(h, w), mode='bilinear', align_corners=align_corners)
                x_dis = F.interpolate(x_dis, size=(h, w), mode='bilinear', align_corners=align_corners)
            y_col = self.attention(x_col)
            y_dis = self.attention_p(x_dis)
            if self.cfg['psa']['compact']:
                y_dis = y_dis.view(n, h * w, h * w).transpose(1, 2).view(n, h * w, h, w)
            else:
                y_col = self.psamask_collect(y_col)
                y_dis = self.psamask_distribute(y_dis)
            if self.cfg['psa']['psa_softmax']:
                y_col = F.softmax(y_col, dim=1)
                y_dis = F.softmax(y_dis, dim=1)
            x_col = torch.bmm(x_col.view(n, c, h * w), y_col.view(n, h * w, h * w)).view(n, c, h, w) * (1.0 / self.cfg['psa']['normalization_factor'])
            x_dis = torch.bmm(x_dis.view(n, c, h * w), y_dis.view(n, h * w, h * w)).view(n, c, h, w) * (1.0 / self.cfg['psa']['normalization_factor'])
            out = torch.cat([x_col, x_dis], 1)
        feats = self.proj(out)
        feats = F.interpolate(feats, size=identity.shape[2:], mode='bilinear', align_corners=align_corners)
        # feed to decoder
        feats = torch.cat([identity, feats], dim=1)
        preds = self.decoder(feats)
        # feed to auxiliary decoder and return according to the mode
        if self.mode == 'TRAIN':
            preds = F.interpolate(preds, size=(h_input, w_input), mode='bilinear', align_corners=self.align_corners)
            preds_aux = self.auxiliary_decoder(x3)
            preds_aux = F.interpolate(preds_aux, size=(h_input, w_input), mode='bilinear', align_corners=self.align_corners)
            return self.calculatelosses(
                predictions={'loss_cls': preds, 'loss_aux': preds_aux}, 
                targets=targets, 
                losses_cfg=losses_cfg
            )
        return preds
    '''return all layers'''
    def alllayers(self):
        layers = {
            'backbone_net': self.backbone_net,
            'reduce': self.reduce,
            'attention': self.attention,
            'proj': self.proj,
            'decoder': self.decoder,
            'auxiliary_decoder': self.auxiliary_decoder
        }
        if hasattr(self, 'reduce_p'): layers['reduce_p'] = self.reduce_p
        if hasattr(self, 'attention_p'): layers['attention_p'] = self.attention_p
        if hasattr(self, 'psamask_collect'): layers['psamask_collect'] = self.psamask_collect
        if hasattr(self, 'psamask_distribute'): layers['psamask_distribute'] = self.psamask_distribute
        if hasattr(self, 'psamask'): layers['psamask'] = self.psamask
        return layers