'''
Function:
    Implementation of PSANet
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import BaseModel
from mmcv.ops import PSAMask
from ...backbones import BuildActivation, BuildNormalization, constructnormcfg


'''PSANet'''
class PSANet(BaseModel):
    def __init__(self, cfg, mode):
        super(PSANet, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg = self.align_corners, self.norm_cfg, self.act_cfg
        # build psa
        psa_cfg = cfg['psa']
        assert psa_cfg['type'] in ['collect', 'distribute', 'bi-direction']
        mask_h, mask_w = psa_cfg['mask_size']
        if 'normalization_factor' not in psa_cfg:
            psa_cfg['normalization_factor'] = mask_h * mask_w
        self.reduce = nn.Sequential(
            nn.Conv2d(psa_cfg['in_channels'], psa_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(constructnormcfg(placeholder=psa_cfg['out_channels'], norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
        )
        self.attention = nn.Sequential(
            nn.Conv2d(psa_cfg['out_channels'], psa_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(constructnormcfg(placeholder=psa_cfg['out_channels'], norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
            nn.Conv2d(psa_cfg['out_channels'], mask_h * mask_w, kernel_size=1, stride=1, padding=0, bias=False),
        )
        if psa_cfg['type'] == 'bi-direction':
            self.reduce_p = nn.Sequential(
                nn.Conv2d(psa_cfg['in_channels'], psa_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False),
                BuildNormalization(constructnormcfg(placeholder=psa_cfg['out_channels'], norm_cfg=norm_cfg)),
                BuildActivation(act_cfg),
            )
            self.attention_p = nn.Sequential(
                nn.Conv2d(psa_cfg['out_channels'], psa_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False),
                BuildNormalization(constructnormcfg(placeholder=psa_cfg['out_channels'], norm_cfg=norm_cfg)),
                BuildActivation(act_cfg),
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
            BuildNormalization(constructnormcfg(placeholder=psa_cfg['in_channels'], norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
        )
        # build decoder
        decoder_cfg = cfg['decoder']
        self.decoder = nn.Sequential(
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(constructnormcfg(placeholder=decoder_cfg['out_channels'], norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
            nn.Dropout2d(decoder_cfg['dropout']),
            nn.Conv2d(decoder_cfg['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        # build auxiliary decoder
        self.setauxiliarydecoder(cfg['auxiliary'])
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
    '''forward'''
    def forward(self, x, targets=None, losses_cfg=None):
        img_size = x.size(2), x.size(3)
        # feed to backbone network
        backbone_outputs = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to psa
        identity = backbone_outputs[-1]
        shrink_factor, align_corners = self.cfg['psa']['shrink_factor'], self.align_corners
        if self.cfg['psa']['type'] in ['collect', 'distribute']:
            out = self.reduce(backbone_outputs[-1])
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
            x_col = self.reduce(backbone_outputs[-1])
            x_dis = self.reduce_p(backbone_outputs[-1])
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
        predictions = self.decoder(feats)
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
    '''return all layers'''
    def alllayers(self):
        all_layers = {
            'backbone_net': self.backbone_net,
            'reduce': self.reduce,
            'attention': self.attention,
            'proj': self.proj,
            'decoder': self.decoder,
        }
        if hasattr(self, 'reduce_p'): all_layers['reduce_p'] = self.reduce_p
        if hasattr(self, 'attention_p'): all_layers['attention_p'] = self.attention_p
        if hasattr(self, 'psamask_collect'): all_layers['psamask_collect'] = self.psamask_collect
        if hasattr(self, 'psamask_distribute'): all_layers['psamask_distribute'] = self.psamask_distribute
        if hasattr(self, 'psamask'): all_layers['psamask'] = self.psamask
        if hasattr(self, 'auxiliary_decoder'): all_layers['auxiliary_decoder'] = self.auxiliary_decoder
        return all_layers