'''
Function:
    Implementation of "MCIBI++: Soft Mining Contextual Information Beyond Image for Semantic Segmentation"
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..deeplabv3 import ASPP
from ..base import BaseSegmentor
from ..base import SelfAttentionBlock
from .memoryv2 import FeaturesMemoryV2
from ..pspnet import PyramidPoolingModule
from ....utils import SSSegOutputStructure
from ...backbones import BuildActivation, BuildNormalization


'''MCIBIPlusPlus'''
class MCIBIPlusPlus(BaseSegmentor):
    def __init__(self, cfg, mode):
        super(MCIBIPlusPlus, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        # build memory
        context_within_image_cfg = head_cfg['context_within_image']
        if context_within_image_cfg['is_on']:
            cwi_cfg = context_within_image_cfg['cfg']
            cwi_cfg.update({
                'in_channels': head_cfg['in_channels'], 'out_channels': head_cfg['feats_channels'], 'align_corners': align_corners,
                'norm_cfg': copy.deepcopy(norm_cfg), 'act_cfg': copy.deepcopy(act_cfg),
            })
            supported_context_modules = {
                'aspp': ASPP, 'ppm': PyramidPoolingModule,
            }
            if context_within_image_cfg['type'] == 'aspp':
                cwi_cfg.pop('pool_scales')
            elif context_within_image_cfg['type'] == 'ppm':
                cwi_cfg.pop('dilations')
            self.context_within_image_module = supported_context_modules[context_within_image_cfg['type']](**cwi_cfg)
            if context_within_image_cfg.get('use_self_attention', True):
                self.self_attention = SelfAttentionBlock(key_in_channels=head_cfg['feats_channels'], query_in_channels=head_cfg['feats_channels'],
                    transform_channels=head_cfg['feats_channels']//2, out_channels=head_cfg['feats_channels'], share_key_query=False, query_downsample=None,
                    key_downsample=None, key_query_num_convs=2, value_out_num_convs=1, key_query_norm=True, value_out_norm=True, matmul_norm=True,
                    with_out_project=True, norm_cfg=norm_cfg, act_cfg=act_cfg,
                )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(head_cfg['in_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        self.memory_module = FeaturesMemoryV2(
            num_classes=cfg['num_classes'], feats_channels=head_cfg['feats_channels'], transform_channels=head_cfg['transform_channels'], out_channels=head_cfg['out_channels'], 
            use_hard_aggregate=head_cfg['use_hard_aggregate'], downsample_before_sa=head_cfg['downsample_before_sa'], norm_cfg=copy.deepcopy(norm_cfg), act_cfg=copy.deepcopy(act_cfg),
            align_corners=align_corners,
        )
        # build fpn
        if head_cfg.get('fpn', None) is not None:
            act_cfg_copy = copy.deepcopy(act_cfg)
            if 'inplace' in act_cfg_copy: act_cfg_copy['inplace'] = False
            self.lateral_convs = nn.ModuleList()
            for in_channels in head_cfg['fpn']['in_channels_list'][:-1]:
                self.lateral_convs.append(nn.Sequential(
                    nn.Conv2d(in_channels, head_cfg['fpn']['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False),
                    BuildNormalization(placeholder=head_cfg['fpn']['feats_channels'], norm_cfg=norm_cfg),
                    BuildActivation(act_cfg_copy),
                ))
            self.fpn_convs = nn.ModuleList()
            for in_channels in [head_cfg['fpn']['feats_channels'],] * len(self.lateral_convs):
                self.fpn_convs.append(nn.Sequential(
                    nn.Conv2d(in_channels, head_cfg['fpn']['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),
                    BuildNormalization(placeholder=head_cfg['fpn']['out_channels'], norm_cfg=norm_cfg),
                    BuildActivation(act_cfg_copy),
                ))
        # build decoder
        for key, value in head_cfg['decoder'].items():
            if key == 'cwi' and (not context_within_image_cfg['is_on']): continue
            setattr(self, f'decoder_{key}', nn.Sequential())
            decoder = getattr(self, f'decoder_{key}')
            decoder.add_module('conv1', nn.Conv2d(value['in_channels'], value['out_channels'], kernel_size=value.get('kernel_size', 1), stride=1, padding=value.get('padding', 0), bias=False))
            decoder.add_module('bn1', BuildNormalization(placeholder=value['out_channels'], norm_cfg=norm_cfg))
            decoder.add_module('act1', BuildActivation(act_cfg))
            decoder.add_module('dropout', nn.Dropout2d(value['dropout']))
            decoder.add_module('conv2', nn.Conv2d(value['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        # build auxiliary decoder
        self.setauxiliarydecoder(cfg['auxiliary'])
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
    '''forward'''
    def forward(self, data_meta, **kwargs):
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        # feed to backbone network
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to context within image module
        if hasattr(self, 'context_within_image_module'):
            feats_cwi = self.context_within_image_module(backbone_outputs[-1])
            if hasattr(self, 'decoder_cwi'): preds_cwi = self.decoder_cwi(feats_cwi)
        # feed to memory
        pixel_representations = self.bottleneck(backbone_outputs[-1])
        preds_pr = self.decoder_pr(pixel_representations)
        if self.cfg['head'].get('force_use_preds_pr', False):
            memory_gather_logits = preds_pr
        else:
            memory_gather_logits = preds_cwi if (hasattr(self, 'context_within_image_module') and hasattr(self, 'decoder_cwi')) else preds_pr
        memory_input = pixel_representations
        assert memory_input.shape[2:] == memory_gather_logits.shape[2:]
        if (self.mode == 'TRAIN') and (kwargs['epoch'] < self.cfg['head'].get('warmup_epoch', 0)):
            with torch.no_grad():
                gt = data_meta.gettargets()['seg_targets']
                gt = F.interpolate(gt.unsqueeze(1), size=memory_gather_logits.shape[2:], mode='nearest')[:, 0, :, :]
                assert len(gt.shape) == 3, 'seg_targets format error'
                preds_gt = gt.new_zeros(memory_gather_logits.shape).type_as(memory_gather_logits)
                valid_mask = (gt >= 0) & (gt < self.cfg['num_classes'])
                idxs = torch.nonzero(valid_mask, as_tuple=True)
                if idxs[0].numel() > 0:
                    preds_gt[idxs[0], gt[valid_mask].long(), idxs[1], idxs[2]] = 1
            stored_memory, memory_output = self.memory_module(memory_input, preds_gt.detach())
        else:
            if 'memory_gather_logits' in kwargs: 
                memory_gather_logits_aux = F.interpolate(kwargs['memory_gather_logits'], size=memory_gather_logits.shape[2:], mode='bilinear', align_corners=self.align_corners)
                weights = kwargs.get('memory_gather_logits_weights', [2, 1.5])
                memory_gather_logits = (memory_gather_logits * weights[0] + memory_gather_logits_aux * weights[1]) / (sum(weights) - 1)
            stored_memory, memory_output = self.memory_module(memory_input, memory_gather_logits)
        # feed to fpn & decoder
        if hasattr(self, 'context_within_image_module'):
            if hasattr(self, 'self_attention'): 
                memory_output = self.self_attention(feats_cwi, memory_output)
            if hasattr(self, 'fpn_convs'):
                inputs = backbone_outputs[:-1]
                lateral_outputs = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
                if self.cfg['head'].get('fuse_memory_cwi_before_fpn', True):
                    lateral_outputs.append(torch.cat([memory_output, feats_cwi], dim=1))
                else:
                    lateral_outputs.append(feats_cwi)
                for i in range(len(lateral_outputs) - 1, 0, -1):
                    prev_shape = lateral_outputs[i - 1].shape[2:]
                    lateral_outputs[i - 1] = lateral_outputs[i - 1] + F.interpolate(lateral_outputs[i], size=prev_shape, mode='bilinear', align_corners=self.align_corners)
                fpn_outputs = [self.fpn_convs[i](lateral_outputs[i]) for i in range(len(lateral_outputs) - 1)]
                fpn_outputs.append(lateral_outputs[-1])
                fpn_outputs = [F.interpolate(out, size=fpn_outputs[0].size()[2:], mode='bilinear', align_corners=self.align_corners) for out in fpn_outputs]
                if not self.cfg['head'].get('fuse_memory_cwi_before_fpn', True): 
                    fpn_outputs.append(F.interpolate(memory_output, size=fpn_outputs[0].shape[2:], mode='bilinear', align_corners=self.align_corners))
                memory_output = torch.cat(fpn_outputs, dim=1)
            else:
                memory_output = torch.cat([memory_output, feats_cwi], dim=1)
        preds_cls = self.decoder_cls(memory_output)
        # forward according to the mode
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            predictions = self.customizepredsandlosses(
                seg_logits=preds_cls, targets=data_meta.gettargets(), backbone_outputs=backbone_outputs, losses_cfg=self.cfg['losses'], img_size=img_size, auto_calc_loss=False,
            )
            preds_cls = predictions.pop('loss_cls')
            preds_pr = F.interpolate(preds_pr, size=img_size, mode='bilinear', align_corners=self.align_corners)
            predictions.update({'loss_pr': preds_pr, 'loss_cls': preds_cls})
            if hasattr(self, 'context_within_image_module') and hasattr(self, 'decoder_cwi'): 
                preds_cwi = F.interpolate(preds_cwi, size=img_size, mode='bilinear', align_corners=self.align_corners)
                predictions.update({'loss_cwi': preds_cwi})
            with torch.no_grad():
                self.memory_module.update(
                    features=F.interpolate(pixel_representations, size=img_size, mode='bilinear', align_corners=self.align_corners), 
                    segmentation=data_meta.gettargets()['seg_targets'], learning_rate=kwargs['learning_rate'], **self.cfg['head']['update_cfg']
                )
            loss, losses_log_dict = self.calculatelosses(
                predictions=predictions, targets=data_meta.gettargets(), losses_cfg=self.cfg['losses'],
            )
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict) if self.mode == 'TRAIN' else SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict, seg_logits=preds_cls)
        else:
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, seg_logits=preds_cls)
        return ssseg_outputs