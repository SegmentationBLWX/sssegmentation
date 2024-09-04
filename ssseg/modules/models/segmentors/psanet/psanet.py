'''
Function:
    Implementation of PSANet
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import BaseSegmentor
from ....utils import SSSegOutputStructure
from ...backbones import BuildActivation, BuildNormalization
try:
    from mmcv.ops import PSAMask
except:
    PSAMask = None


'''PSANet'''
class PSANet(BaseSegmentor):
    def __init__(self, cfg, mode):
        super(PSANet, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        # build psa
        assert head_cfg['type'] in ['collect', 'distribute', 'bi-direction']
        mask_h, mask_w = head_cfg['mask_size']
        if 'normalization_factor' not in self.cfg['head']:
            self.cfg['head']['normalization_factor'] = mask_h * mask_w
        self.reduce = nn.Sequential(
            nn.Conv2d(head_cfg['in_channels'], head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        self.attention = nn.Sequential(
            nn.Conv2d(head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
            nn.Conv2d(head_cfg['feats_channels'], mask_h * mask_w, kernel_size=1, stride=1, padding=0, bias=False),
        )
        if head_cfg['type'] == 'bi-direction':
            self.reduce_p = nn.Sequential(
                nn.Conv2d(head_cfg['in_channels'], head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False),
                BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
                BuildActivation(act_cfg),
            )
            self.attention_p = nn.Sequential(
                nn.Conv2d(head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False),
                BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
                BuildActivation(act_cfg),
                nn.Conv2d(head_cfg['feats_channels'], mask_h * mask_w, kernel_size=1, stride=1, padding=0, bias=False),
            )
            if not head_cfg['compact']:
                self.psamask_collect = PSAMask('collect', head_cfg['mask_size'])
                self.psamask_distribute = PSAMask('distribute', head_cfg['mask_size'])
        else:
            if not head_cfg['compact']:
                self.psamask = PSAMask(head_cfg['type'], head_cfg['mask_size'])
        self.proj = nn.Sequential(
            nn.Conv2d(head_cfg['feats_channels'] * (2 if head_cfg['type'] == 'bi-direction' else 1), head_cfg['in_channels'], kernel_size=1, stride=1, padding=1, bias=False),
            BuildNormalization(placeholder=head_cfg['in_channels'], norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        # build decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(head_cfg['in_channels'] * 2, head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
            nn.Dropout2d(head_cfg['dropout']),
            nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        # build auxiliary decoder
        self.setauxiliarydecoder(cfg['auxiliary'])
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
    '''forward'''
    def forward(self, data_meta):
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        # feed to backbone network
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to psa
        identity = backbone_outputs[-1]
        shrink_factor, align_corners = self.cfg['head']['shrink_factor'], self.align_corners
        if self.cfg['head']['type'] in ['collect', 'distribute']:
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
            if self.cfg['head']['compact']:
                if self.cfg['head']['type'] == 'collect':
                    y = y.view(n, h * w, h * w).transpose(1, 2).view(n, h * w, h, w)
            else:
                y = self.psamask(y)
            if self.cfg['head']['psa_softmax']:
                y = F.softmax(y, dim=1)
            out = torch.bmm(out.view(n, c, h * w), y.view(n, h * w, h * w)).view(n, c, h, w) * (1.0 / self.cfg['head']['normalization_factor'])
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
            if self.cfg['head']['compact']:
                y_dis = y_dis.view(n, h * w, h * w).transpose(1, 2).view(n, h * w, h, w)
            else:
                y_col = self.psamask_collect(y_col)
                y_dis = self.psamask_distribute(y_dis)
            if self.cfg['head']['psa_softmax']:
                y_col = F.softmax(y_col, dim=1)
                y_dis = F.softmax(y_dis, dim=1)
            x_col = torch.bmm(x_col.view(n, c, h * w), y_col.view(n, h * w, h * w)).view(n, c, h, w) * (1.0 / self.cfg['head']['normalization_factor'])
            x_dis = torch.bmm(x_dis.view(n, c, h * w), y_dis.view(n, h * w, h * w)).view(n, c, h, w) * (1.0 / self.cfg['head']['normalization_factor'])
            out = torch.cat([x_col, x_dis], 1)
        feats = self.proj(out)
        feats = F.interpolate(feats, size=identity.shape[2:], mode='bilinear', align_corners=align_corners)
        # feed to decoder
        feats = torch.cat([identity, feats], dim=1)
        seg_logits = self.decoder(feats)
        # forward according to the mode
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            loss, losses_log_dict = self.customizepredsandlosses(
                seg_logits=seg_logits, targets=data_meta.gettargets(), backbone_outputs=backbone_outputs, losses_cfg=self.cfg['losses'], img_size=img_size,
            )
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict) if self.mode == 'TRAIN' else SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict, seg_logits=seg_logits)
        else:
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, seg_logits=seg_logits)
        return ssseg_outputs