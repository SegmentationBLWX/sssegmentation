'''
Function:
    Implementation of SETR
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mla import MLANeck
from ..base import BaseSegmentor
from ....utils import SSSegOutputStructure
from ...backbones import BuildActivation, BuildNormalization


'''Naive upsampling head and Progressive upsampling head of SETR'''
class SETRUP(BaseSegmentor):
    def __init__(self, cfg, mode):
        super(SETRUP, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        # build norm layer
        self.norm_layers = nn.ModuleList()
        for in_channels in head_cfg['in_channels_list']:
            norm_cfg_copy = head_cfg['norm_cfg'].copy()
            norm_layer = BuildNormalization(placeholder=in_channels, norm_cfg=norm_cfg_copy)
            self.norm_layers.append(norm_layer)
        # build decoder
        self.decoder = self.builddecoder({
            'in_channels': head_cfg['in_channels_list'][-1], 'out_channels': head_cfg['feats_channels'], 'kernel_size': head_cfg['kernel_size'],
            'scale_factor': head_cfg['scale_factor'], 'dropout': head_cfg['dropout'], 'num_convs': head_cfg['num_convs'],
        })
        # build auxiliary decoder
        auxiliary_cfg_list = cfg['auxiliary']
        assert isinstance(auxiliary_cfg_list, (tuple, list))
        self.auxiliary_decoders = nn.ModuleList()
        for auxiliary_cfg in auxiliary_cfg_list:
            decoder = self.builddecoder(auxiliary_cfg)
            self.auxiliary_decoders.append(decoder)
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
    '''forward'''
    def forward(self, data_meta):
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        # feed to backbone network
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to norm layer
        assert len(backbone_outputs) == len(self.norm_layers)
        for idx in range(len(backbone_outputs)):
            backbone_outputs[idx] = self.norm(backbone_outputs[idx], self.norm_layers[idx])
        # feed to decoder
        seg_logits = self.decoder(backbone_outputs[-1])
        # forward according to the mode
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            seg_logits = F.interpolate(seg_logits, size=img_size, mode='bilinear', align_corners=self.align_corners)
            predictions = {'loss_cls': seg_logits}
            backbone_outputs = backbone_outputs[:-1]
            for idx, (out, dec) in enumerate(zip(backbone_outputs, self.auxiliary_decoders)):
                seg_logits_aux = dec(out)
                seg_logits_aux = F.interpolate(seg_logits_aux, size=img_size, mode='bilinear', align_corners=self.align_corners)
                predictions[f'loss_aux{idx+1}'] = seg_logits_aux
            loss, losses_log_dict = self.calculatelosses(predictions=predictions, targets=data_meta.gettargets(), losses_cfg=self.cfg['losses'])
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict) if self.mode == 'TRAIN' else SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict, seg_logits=seg_logits)
        else:
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, seg_logits=seg_logits)
        return ssseg_outputs
    '''norm layer'''
    def norm(self, x, norm_layer):
        n, c, h, w = x.shape
        x = x.reshape(n, c, h * w).transpose(2, 1).contiguous()
        x = norm_layer(x)
        x = x.transpose(1, 2).reshape(n, c, h, w).contiguous()
        return x
    '''build decoder'''
    def builddecoder(self, decoder_cfg):
        layers, norm_cfg, act_cfg, num_classes, align_corners, kernel_size = [], self.norm_cfg.copy(), self.act_cfg.copy(), self.cfg['num_classes'], self.align_corners, decoder_cfg['kernel_size']
        for idx in range(decoder_cfg['num_convs']):
            if idx == 0:
                layers.append(nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['out_channels'], kernel_size=kernel_size, stride=1, padding=int(kernel_size - 1) // 2, bias=False))
            else:
                layers.append(nn.Conv2d(decoder_cfg['out_channels'], decoder_cfg['out_channels'], kernel_size=kernel_size, stride=1, padding=int(kernel_size - 1) // 2, bias=False))
            layers.append(BuildNormalization(placeholder=decoder_cfg['out_channels'], norm_cfg=norm_cfg))
            layers.append(BuildActivation(act_cfg))
            layers.append(nn.Upsample(scale_factor=decoder_cfg['scale_factor'], mode='bilinear', align_corners=align_corners))
        layers.append(nn.Dropout2d(decoder_cfg['dropout']))
        layers.append(nn.Conv2d(decoder_cfg['out_channels'], num_classes, kernel_size=1, stride=1, padding=0))
        return nn.Sequential(*layers)


'''Multi level feature aggretation head of SETR'''
class SETRMLA(BaseSegmentor):
    def __init__(self, cfg, mode):
        super(SETRMLA, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        # build mla neck
        norm_layers = nn.ModuleList()
        for in_channels in head_cfg['in_channels_list']:
            norm_cfg_copy = head_cfg['norm_cfg'].copy()
            norm_layer = BuildNormalization(placeholder=in_channels, norm_cfg=norm_cfg_copy)
            norm_layers.append(norm_layer)
        self.mla_neck = MLANeck(
            in_channels_list=head_cfg['in_channels_list'], out_channels=head_cfg['mla_feats_channels'], norm_layers=norm_layers, norm_cfg=norm_cfg, act_cfg=act_cfg,
        )
        # build upsample convs and decoder
        assert head_cfg['mla_up_channels'] * len(head_cfg['in_channels_list']) == head_cfg['feats_channels']
        self.up_convs = nn.ModuleList()
        for i in range(len(head_cfg['in_channels_list'])):
            self.up_convs.append(nn.Sequential(
                nn.Conv2d(head_cfg['mla_feats_channels'], head_cfg['mla_up_channels'], kernel_size=3, stride=1, padding=1, bias=False),
                BuildNormalization(placeholder=head_cfg['mla_up_channels'], norm_cfg=norm_cfg),
                BuildActivation(act_cfg),
                nn.Conv2d(head_cfg['mla_up_channels'], head_cfg['mla_up_channels'], kernel_size=3, stride=1, padding=1, bias=False),
                BuildNormalization(placeholder=head_cfg['mla_up_channels'], norm_cfg=norm_cfg),
                BuildActivation(act_cfg),
                nn.Upsample(scale_factor=head_cfg['scale_factor'], mode='bilinear', align_corners=align_corners)
            ))
        self.decoder = nn.Sequential(
            nn.Dropout2d(head_cfg['dropout']),
            nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0),
        )
        # build auxiliary decoder
        auxiliary_cfg_list = cfg['auxiliary']
        assert isinstance(auxiliary_cfg_list, (tuple, list))
        self.auxiliary_decoders = nn.ModuleList()
        for auxiliary_cfg in auxiliary_cfg_list:
            decoder = self.builddecoder(auxiliary_cfg)
            self.auxiliary_decoders.append(decoder)
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
    '''forward'''
    def forward(self, data_meta):
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        # feed to backbone network
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to mla neck
        feats_list = self.mla_neck(list(backbone_outputs))
        # feed to decoder
        feats_outputs = []
        assert len(feats_list) == len(self.up_convs)
        for feats, up_conv in zip(feats_list, self.up_convs):
            feats_outputs.append(up_conv(feats))
        feats_outputs = torch.cat(feats_outputs, dim=1)
        seg_logits = self.decoder(feats_outputs)
        # forward according to the mode
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            seg_logits = F.interpolate(seg_logits, size=img_size, mode='bilinear', align_corners=self.align_corners)
            predictions = {'loss_cls': seg_logits}
            feats_list = feats_list[-len(self.auxiliary_decoders):]
            for idx, (out, dec) in enumerate(zip(feats_list, self.auxiliary_decoders)):
                seg_logits_aux = dec(out)
                seg_logits_aux = F.interpolate(seg_logits_aux, size=img_size, mode='bilinear', align_corners=self.align_corners)
                predictions[f'loss_aux{idx+1}'] = seg_logits_aux
            loss, losses_log_dict = self.calculatelosses(predictions=predictions, targets=data_meta.gettargets(), losses_cfg=self.cfg['losses'])
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict) if self.mode == 'TRAIN' else SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict, seg_logits=seg_logits)
        else:
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, seg_logits=seg_logits)
        return ssseg_outputs
    '''build decoder'''
    def builddecoder(self, decoder_cfg):
        layers, norm_cfg, act_cfg, num_classes, align_corners, kernel_size = [], self.norm_cfg.copy(), self.act_cfg.copy(), self.cfg['num_classes'], self.align_corners, decoder_cfg['kernel_size']
        for idx in range(decoder_cfg['num_convs']):
            if idx == 0:
                layers.append(nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['out_channels'], kernel_size=kernel_size, stride=1, padding=int(kernel_size - 1) // 2, bias=False))
            else:
                layers.append(nn.Conv2d(decoder_cfg['out_channels'], decoder_cfg['out_channels'], kernel_size=kernel_size, stride=1, padding=int(kernel_size - 1) // 2, bias=False))
            layers.append(BuildNormalization(placeholder=decoder_cfg['out_channels'], norm_cfg=norm_cfg))
            layers.append(BuildActivation(act_cfg))
            layers.append(nn.Upsample(scale_factor=decoder_cfg['scale_factor'], mode='bilinear', align_corners=align_corners))
        layers.append(nn.Dropout2d(decoder_cfg['dropout']))
        layers.append(nn.Conv2d(decoder_cfg['out_channels'], num_classes, kernel_size=1, stride=1, padding=0))
        return nn.Sequential(*layers)