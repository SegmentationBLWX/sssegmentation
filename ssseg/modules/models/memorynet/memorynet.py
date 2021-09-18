'''
Function:
    Implementation of MemoryNet - Mining Contextual Information Beyond Image for Semantic Segmentation
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from ...backbones import *
from ..base import BaseModel
from ..deeplabv3 import ASPP
from .memory import FeaturesMemory
from ..pspnet import PyramidPoolingModule


'''MemoryNet'''
class MemoryNet(BaseModel):
    def __init__(self, cfg, **kwargs):
        super(MemoryNet, self).__init__(cfg, **kwargs)
        align_corners, norm_cfg, act_cfg = self.align_corners, self.norm_cfg, self.act_cfg
        # build norm layer
        if 'normlayer' in cfg:
            self.norm_layers = nn.ModuleList()
            for in_channels in cfg['normlayer']['in_channels_list']:
                norm_layer = BuildNormalization(cfg['normlayer']['type'], (in_channels, cfg['normlayer']['opts']))
                self.norm_layers.append(norm_layer)
        # build memory
        memory_cfg = cfg['memory']
        if memory_cfg['downsample_backbone']['stride'] > 1:
            self.downsample_backbone = nn.Sequential(
                nn.Conv2d(memory_cfg['in_channels'], memory_cfg['in_channels'], **memory_cfg['downsample_backbone']),
                BuildNormalization(norm_cfg['type'], (memory_cfg['in_channels'], norm_cfg['opts'])),
                BuildActivation(act_cfg['type'], **act_cfg['opts']),
            )
        context_within_image_cfg = memory_cfg['context_within_image']
        if context_within_image_cfg['is_on']:
            cwi_cfg = context_within_image_cfg['cfg']
            cwi_cfg.update({
                'in_channels': memory_cfg['in_channels'],
                'out_channels': memory_cfg['feats_channels'],
                'align_corners': align_corners,
                'norm_cfg': copy.deepcopy(norm_cfg),
                'act_cfg': copy.deepcopy(act_cfg),
            })
            supported_context_modules = {
                'aspp': ASPP,
                'ppm': PyramidPoolingModule,
            }
            self.context_within_image_module = supported_context_modules[context_within_image_cfg['type']](**cwi_cfg)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(memory_cfg['in_channels'], memory_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(norm_cfg['type'], (memory_cfg['feats_channels'], norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts']),
        )
        self.memory_module = FeaturesMemory(
            num_classes=cfg['num_classes'], 
            feats_channels=memory_cfg['feats_channels'], 
            transform_channels=memory_cfg['transform_channels'],
            num_feats_per_cls=memory_cfg['num_feats_per_cls'],
            out_channels=memory_cfg['out_channels'],
            use_context_within_image=context_within_image_cfg['is_on'],
            use_hard_aggregate=memory_cfg['use_hard_aggregate'],
            norm_cfg=copy.deepcopy(norm_cfg),
            act_cfg=copy.deepcopy(act_cfg),
        )
        # build decoder
        decoder_cfg = cfg['decoder']['stage1']
        self.decoder_stage1 = nn.Sequential(
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(norm_cfg['type'], (decoder_cfg['out_channels'], norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts']),
            nn.Dropout2d(decoder_cfg['dropout']),
            nn.Conv2d(decoder_cfg['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0),
        )
        decoder_cfg = cfg['decoder']['stage2']
        self.decoder_stage2 = nn.Sequential(
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(norm_cfg['type'], (decoder_cfg['out_channels'], norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts']),
            nn.Dropout2d(decoder_cfg['dropout']),
            nn.Conv2d(decoder_cfg['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        # build auxiliary decoder
        auxiliary_cfg = cfg['auxiliary']
        if auxiliary_cfg is not None:
            assert type(auxiliary_cfg) in [dict, list], 'auxiliary_cfg parse error...'
            if isinstance(auxiliary_cfg, list):
                self.auxiliary_decoder = nn.ModuleList()
                for aux_cfg in auxiliary_cfg:
                    decoder = nn.Sequential(
                        nn.Conv2d(aux_cfg['in_channels'], aux_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),
                        BuildNormalization(norm_cfg['type'], (aux_cfg['out_channels'], norm_cfg['opts'])),
                        BuildActivation(act_cfg['type'], **act_cfg['opts']),
                        nn.Upsample(scale_factor=4),
                        nn.Conv2d(aux_cfg['out_channels'], aux_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),
                        BuildNormalization(norm_cfg['type'], (aux_cfg['out_channels'], norm_cfg['opts'])),
                        BuildActivation(act_cfg['type'], **act_cfg['opts']),
                        nn.Upsample(scale_factor=4),
                        nn.Dropout2d(aux_cfg['dropout']),
                        nn.Conv2d(aux_cfg['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
                    )
                    self.auxiliary_decoder.append(decoder)
            else:
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
    def forward(self, x, targets=None, losses_cfg=None, **kwargs):
        h, w = x.size(2), x.size(3)
        # feed to backbone network
        x1, x2, x3, x4 = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))
        if hasattr(self, 'norm_layers'):
            x1 = self.norm(x1, self.norm_layers[0])
            x2 = self.norm(x2, self.norm_layers[1])
            x3 = self.norm(x3, self.norm_layers[2])
            x4 = self.norm(x4, self.norm_layers[3])
        if self.cfg['memory']['downsample_backbone']['stride'] > 1:
            x1, x2, x3, x4 = self.downsample_backbone(x1), self.downsample_backbone(x2), self.downsample_backbone(x3), self.downsample_backbone(x4)
        # feed to context within image module
        feats_ms = self.context_within_image_module(x4) if hasattr(self, 'context_within_image_module') else None
        # feed to memory
        memory_input = self.bottleneck(x4)
        preds_stage1 = self.decoder_stage1(memory_input)
        stored_memory, memory_output = self.memory_module(memory_input, preds_stage1, feats_ms)
        # feed to decoder
        preds_stage2 = self.decoder_stage2(memory_output)
        # feed to auxiliary decoder and return according to the mode
        if self.mode == 'TRAIN':
            preds_stage1 = F.interpolate(preds_stage1, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            preds_stage2 = F.interpolate(preds_stage2, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            if hasattr(self, 'auxiliary_decoder'):
                if isinstance(self.cfg['auxiliary'], list):
                    preds_aux1 = self.auxiliary_decoder[0](x1)
                    preds_aux2 = self.auxiliary_decoder[1](x2)
                    preds_aux3 = self.auxiliary_decoder[2](x3)
                    preds_aux1 = F.interpolate(preds_aux1, size=(h, w), mode='bilinear', align_corners=self.align_corners)
                    preds_aux2 = F.interpolate(preds_aux2, size=(h, w), mode='bilinear', align_corners=self.align_corners)
                    preds_aux3 = F.interpolate(preds_aux3, size=(h, w), mode='bilinear', align_corners=self.align_corners)
                    outputs_dict = {
                        'loss_cls_stage1': preds_stage1, 
                        'loss_cls_stage2': preds_stage2, 
                        'loss_aux1': preds_aux1,
                        'loss_aux2': preds_aux2,
                        'loss_aux3': preds_aux3,
                    }
                else:
                    preds_aux = self.auxiliary_decoder(x3)
                    preds_aux = F.interpolate(preds_aux, size=(h, w), mode='bilinear', align_corners=self.align_corners)
                    outputs_dict = {
                        'loss_cls_stage1': preds_stage1, 
                        'loss_cls_stage2': preds_stage2, 
                        'loss_aux': preds_aux
                    }
            else:
                outputs_dict = {
                    'loss_cls_stage1': preds_stage1, 
                    'loss_cls_stage2': preds_stage2, 
                }
            with torch.no_grad():
                self.memory_module.update(
                    features=F.interpolate(memory_input, size=(h, w), mode='bilinear', align_corners=self.align_corners), 
                    segmentation=targets['segmentation'],
                    learning_rate=kwargs['learning_rate'],
                    **self.cfg['memory']['update_cfg']
                )
            loss, losses_log_dict = self.calculatelosses(
                predictions=outputs_dict, 
                targets=targets, 
                losses_cfg=losses_cfg
            )
            if (kwargs['epoch'] > 1) and self.cfg['memory']['use_loss']:
                loss_memory, loss_memory_log = self.calculatememoryloss(stored_memory)
                loss += loss_memory
                losses_log_dict['loss_memory'] = loss_memory_log
                total = losses_log_dict.pop('total') + losses_log_dict['loss_memory']
                losses_log_dict['total'] = total
            return loss, losses_log_dict
        return preds_stage2
    '''norm layer'''
    def norm(self, x, norm_layer):
        n, c, h, w = x.shape
        x = x.reshape(n, c, h * w).transpose(2, 1).contiguous()
        x = norm_layer(x)
        x = x.transpose(1, 2).reshape(n, c, h, w).contiguous()
        return x
    '''calculate memory loss'''
    def calculatememoryloss(self, stored_memory):
        num_classes, num_feats_per_cls, feats_channels = stored_memory.size()
        stored_memory = stored_memory.reshape(num_classes * num_feats_per_cls, feats_channels, 1, 1)
        preds_memory = self.decoder_stage2(stored_memory)
        target = torch.range(0, num_classes - 1).type_as(stored_memory).long()
        target = target.unsqueeze(1).repeat(1, num_feats_per_cls).view(-1)
        loss_memory = self.calculateloss(preds_memory, target, self.cfg['memory']['loss_cfg'])
        if dist.is_available() and dist.is_initialized():
            value = loss_memory.data.clone()
            dist.all_reduce(value.div_(dist.get_world_size()))
        else:
            value = torch.Tensor([loss_memory.item()]).type_as(stored_memory)
        return loss_memory, value
    '''return all layers'''
    def alllayers(self):
        all_layers = {
            'backbone_net': self.backbone_net,
            'bottleneck': self.bottleneck,
            'memory_module': self.memory_module,
            'decoder_stage1': self.decoder_stage1,
            'decoder_stage2': self.decoder_stage2,
        }
        if hasattr(self, 'norm_layers'):
            all_layers.update({
                'norm_layers': self.norm_layers
            })
        if hasattr(self, 'downsample_backbone'):
            all_layers.update({
                'downsample_backbone': self.downsample_backbone
            })
        if hasattr(self, 'context_within_image_module'):
            all_layers.update({
                'context_within_image_module': self.context_within_image_module
            })
        if hasattr(self, 'auxiliary_decoder'):
            all_layers.update({
                'auxiliary_decoder': self.auxiliary_decoder
            })
        return all_layers