'''
Function:
    Implementation of IDRNet
Author:
    Zhenchao Jin
'''
import copy
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from ..deeplabv3 import ASPP
from ..pspnet import PyramidPoolingModule
from ....utils import SSSegOutputStructure
from ..base import BaseSegmentor, SelfAttentionBlock
from ...backbones import BuildActivation, BuildNormalization


'''IDRNet'''
class IDRNet(BaseSegmentor):
    def __init__(self, cfg, mode):
        super(IDRNet, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        # build bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(head_cfg['in_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        # coarse context
        if 'coarse_context' in head_cfg:
            supported_coarse_contexts = {
                'aspp': ASPP, 'ppm': PyramidPoolingModule,
            }
            coarse_context_cfg = {
                'in_channels': head_cfg['feats_channels'], 'out_channels': head_cfg['feats_channels'], 
                'align_corners': align_corners, 'norm_cfg': norm_cfg, 'act_cfg': act_cfg
            }
            coarse_context_cfg.update(head_cfg['coarse_context'])
            coarse_context_type = coarse_context_cfg.pop('type')
            if 'fpn' in head_cfg:
                coarse_context_cfg['out_channels'] = head_cfg['fpn']['feats_channels']
            self.coarse_context_module = supported_coarse_contexts[coarse_context_type](**coarse_context_cfg)
            if head_cfg['use_sa_on_coarsecontext_before']:
                self.coarsecontext_refiner_before = SelfAttentionBlock(
                    key_in_channels=coarse_context_cfg['out_channels'], query_in_channels=coarse_context_cfg['out_channels'], transform_channels=head_cfg['refine_coarsecontext_channels'],
                    out_channels=coarse_context_cfg['out_channels'], share_key_query=False, query_downsample=None, key_downsample=None, key_query_num_convs=2,
                    value_out_num_convs=1, key_query_norm=True, value_out_norm=True, matmul_norm=True, with_out_project=True, norm_cfg=norm_cfg, act_cfg=act_cfg
                )
            elif head_cfg['use_sa_on_coarsecontext_after']:
                self.coarsecontext_refiner_after = SelfAttentionBlock(
                    key_in_channels=coarse_context_cfg['out_channels'], query_in_channels=coarse_context_cfg['out_channels'], transform_channels=head_cfg['refine_coarsecontext_channels'],
                    out_channels=coarse_context_cfg['out_channels'], share_key_query=False, query_downsample=None, key_downsample=None, key_query_num_convs=2,
                    value_out_num_convs=1, key_query_norm=True, value_out_norm=True, matmul_norm=True, with_out_project=True, norm_cfg=norm_cfg, act_cfg=act_cfg
                )
        # fpn
        if 'fpn' in head_cfg:
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
        # class relations
        for name in ['class_relations_mean', 'class_relations_var']:
            value = nn.Parameter(torch.eye(cfg['num_classes']).float(), requires_grad=False)
            setattr(self, name, value)
        self.selected_classes_counter = nn.Parameter(
            torch.ones(cfg['num_classes']).float() * 1e-6, requires_grad=False
        )
        # idcontext refiner
        self.idcontext_refiner = SelfAttentionBlock(
            key_in_channels=head_cfg['feats_channels'] * 6, query_in_channels=head_cfg['feats_channels'] * 6, transform_channels=head_cfg['refine_idcontext_channels'],
            out_channels=head_cfg['feats_channels'], share_key_query=False, query_downsample=None, key_downsample=None, key_query_num_convs=2,
            value_out_num_convs=1, key_query_norm=True, value_out_norm=True, matmul_norm=True, with_out_project=True, norm_cfg=norm_cfg, act_cfg=act_cfg
        )
        # dataset-level class representations
        self.dl_cls_representations = nn.Parameter(
            torch.zeros(cfg['num_classes'], head_cfg['feats_channels']).float(), requires_grad=False
        )
        # build decoder
        if hasattr(self, 'coarse_context_module') and ('fpn' in head_cfg) and head_cfg['use_fpn_before']:
            decoder_stage1_in_channels = coarse_context_cfg['out_channels'] + head_cfg['fpn']['out_channels'] * 3
        else:
            decoder_stage1_in_channels = coarse_context_cfg['out_channels'] if 'coarse_context' in head_cfg else head_cfg['feats_channels']
        if head_cfg['force_stage1_use_oripr']:
            decoder_stage1_in_channels = head_cfg['feats_channels']
        if not hasattr(self, 'coarse_context_module'):
            decoder_stage2_in_channels = head_cfg['feats_channels'] * 2
        elif hasattr(self, 'coarse_context_module') and 'fpn' not in head_cfg:
            decoder_stage2_in_channels = head_cfg['feats_channels'] * 2 + coarse_context_cfg['out_channels']
        elif hasattr(self, 'coarse_context_module') and 'fpn' in head_cfg:
            decoder_stage2_in_channels = head_cfg['feats_channels'] * 2 + coarse_context_cfg['out_channels'] + head_cfg['fpn']['out_channels'] * 3
        for (name, in_channels) in [('decoder_stage1', decoder_stage1_in_channels), ('decoder_stage2', decoder_stage2_in_channels)]:
            value = nn.Sequential(
                nn.Conv2d(in_channels, head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False),
                BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
                BuildActivation(act_cfg),
                nn.Dropout2d(head_cfg['dropout']),
                nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0),
            )
            setattr(self, name, value)
        # build auxiliary decoder
        self.setauxiliarydecoder(cfg['auxiliary'])
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
    '''forward'''
    def forward(self, data_meta):
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        seed = random.randint(1, 1e16)
        # feed to backbone network
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to the bottleneck
        feats, coarse_context = self.bottleneck(backbone_outputs[-1]), None
        # feed to coarse context module and decoder_stage1
        if hasattr(self, 'coarse_context_module'):
            coarse_context = self.coarse_context_module(feats)
            if hasattr(self, 'coarsecontext_refiner_before'):
                assert not hasattr(self, 'coarsecontext_refiner_after')
                coarse_context = self.coarsecontext_refiner_before(coarse_context, coarse_context)
        if hasattr(self, 'fpn_convs') and self.cfg['head']['use_fpn_before']:
            assert not self.cfg['head']['use_fpn_after']
            assert coarse_context is not None, 'upernet setting error'
            inputs = backbone_outputs[:-1]
            lateral_outputs = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
            lateral_outputs.append(coarse_context)
            for i in range(len(lateral_outputs) - 1, 0, -1):
                prev_shape = lateral_outputs[i - 1].shape[2:]
                lateral_outputs[i - 1] = lateral_outputs[i - 1] + F.interpolate(lateral_outputs[i], size=prev_shape, mode='bilinear', align_corners=self.align_corners)
            fpn_outputs = [self.fpn_convs[i](lateral_outputs[i]) for i in range(len(lateral_outputs) - 1)]
            fpn_outputs.append(lateral_outputs[-1])
            fpn_outputs = [F.interpolate(out, size=fpn_outputs[0].size()[2:], mode='bilinear', align_corners=self.align_corners) for out in fpn_outputs]
            coarse_context = torch.cat(fpn_outputs, dim=1)
        if self.cfg['head']['force_stage1_use_oripr']:
            preds_stage1 = self.decoder_stage1(feats)
        else:
            preds_stage1 = self.decoder_stage1(feats if coarse_context is None else coarse_context)
        if preds_stage1.shape[2:] != feats.shape[2:]:
            preds_stage1 = F.interpolate(preds_stage1, size=feats.shape[2:], mode='bilinear', align_corners=self.align_corners)
        if hasattr(self, 'coarse_context_module') and hasattr(self, 'coarsecontext_refiner_after'):
            assert not hasattr(self, 'coarsecontext_refiner_before')
            coarse_context = self.coarsecontext_refiner_after(coarse_context, coarse_context)
        # insert dl_cls_representations into feats
        feats_withdl = self.insertdlrepresentations(feats, preds_stage1)
        # obtain intervention-driven contextual information
        id_context_mean, valid_clsids_batch = self.obtainidcontext(feats_withdl, preds_stage1, self.class_relations_mean)
        id_context_var, _ = self.obtainidcontext(feats_withdl, preds_stage1, self.class_relations_var, None, False)
        id_context = self.idcontext_refiner(torch.cat([feats_withdl, id_context_mean, id_context_var], dim=1), torch.cat([feats_withdl, id_context_mean, id_context_var], dim=1))
        # feed to decoder_stage2
        if hasattr(self, 'fpn_convs') and self.cfg['head']['use_fpn_after']:
            assert not self.cfg['head']['use_fpn_before']
            assert coarse_context is not None, 'upernet setting error'
            inputs = backbone_outputs[:-1]
            lateral_outputs = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
            lateral_outputs.append(coarse_context)
            for i in range(len(lateral_outputs) - 1, 0, -1):
                prev_shape = lateral_outputs[i - 1].shape[2:]
                lateral_outputs[i - 1] = lateral_outputs[i - 1] + F.interpolate(lateral_outputs[i], size=prev_shape, mode='bilinear', align_corners=self.align_corners)
            fpn_outputs = [self.fpn_convs[i](lateral_outputs[i]) for i in range(len(lateral_outputs) - 1)]
            fpn_outputs.append(lateral_outputs[-1])
            fpn_outputs = [F.interpolate(out, size=fpn_outputs[0].size()[2:], mode='bilinear', align_corners=self.align_corners) for out in fpn_outputs]
            coarse_context = torch.cat(fpn_outputs, dim=1)
        torch.cuda.manual_seed(seed)
        if (coarse_context is not None) and (feats.shape[2:] != coarse_context.shape[2:]):
            preds_stage2 = self.decoder_stage2(
                torch.cat([feats, id_context] if coarse_context is None else [
                    F.interpolate(feats, size=coarse_context.size()[2:], mode='bilinear', align_corners=self.align_corners), 
                    F.interpolate(id_context, size=coarse_context.size()[2:], mode='bilinear', align_corners=self.align_corners), 
                    coarse_context
                ], dim=1)
            )
        else:
            preds_stage2 = self.decoder_stage2(torch.cat([feats, id_context] if coarse_context is None else [feats, id_context, coarse_context], dim=1))
        # forward according to the mode
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            # --statistical inference
            with torch.no_grad():
                # ----select intervention clsids
                intervention_clsids = []
                for batch_idx in range(feats.shape[0]):
                    valid_clsids = valid_clsids_batch[batch_idx]
                    choice_weights = []
                    for intervention_clsid in valid_clsids:
                        choice_weights.append(1.0 / float(self.selected_classes_counter.data[intervention_clsid].item()))
                    choice_weights = np.array(choice_weights) / sum(choice_weights)
                    intervention_clsid = random.choices(valid_clsids, weights=choice_weights, k=1)[0]
                    intervention_clsids.append(intervention_clsid)
                    self.selected_classes_counter.data[intervention_clsid] = self.selected_classes_counter.data[intervention_clsid] + 1.0
                # ----update class_relations
                momentum = self.cfg['head']['clsrelation_momentum']
                id_context_mean, _ = self.obtainidcontext(feats_withdl, preds_stage1, self.class_relations_mean, intervention_clsids)
                id_context_var, _ = self.obtainidcontext(feats_withdl, preds_stage1, self.class_relations_var, intervention_clsids, False)
                id_context = self.idcontext_refiner(torch.cat([feats_withdl, id_context_mean, id_context_var], dim=1), torch.cat([feats_withdl, id_context_mean, id_context_var], dim=1))
                torch.cuda.manual_seed(seed)
                if (coarse_context is not None) and (feats.shape[2:] != coarse_context.shape[2:]):
                    preds_intervention_stage2 = self.decoder_stage2(
                        torch.cat([feats, id_context] if coarse_context is None else [
                            F.interpolate(feats, size=coarse_context.size()[2:], mode='bilinear', align_corners=self.align_corners), 
                            F.interpolate(id_context, size=coarse_context.size()[2:], mode='bilinear', align_corners=self.align_corners), 
                            coarse_context
                        ], dim=1)
                    )
                else:
                    preds_intervention_stage2 = self.decoder_stage2(torch.cat([feats, id_context] if coarse_context is None else [feats, id_context, coarse_context], dim=1))
                preds_intervention_stage2 = F.interpolate(preds_intervention_stage2, size=img_size, mode='bilinear', align_corners=self.align_corners)
                preds_intervention_stage2 = preds_intervention_stage2.permute(0, 2, 3, 1).contiguous()
                preds_anchor_stage2 = F.interpolate(preds_stage2, size=img_size, mode='bilinear', align_corners=self.align_corners)
                preds_anchor_stage2 = preds_anchor_stage2.permute(0, 2, 3, 1).contiguous()
                for batch_idx in range(feats.shape[0]):
                    gts_iter = data_meta.gettargets()['seg_targets'][batch_idx]
                    clsids = data_meta.gettargets()['seg_targets'][batch_idx].unique()
                    logits_intervention_stage2_iter, logits_anchor_stage2_iter = preds_intervention_stage2[batch_idx], preds_anchor_stage2[batch_idx]
                    for clsid in clsids:
                        clsid = int(clsid.item())
                        if clsid == self.cfg['head']['ignore_index']: continue
                        gts_iter_cls = gts_iter[gts_iter == clsid].long()
                        loss_intervention_stage2 = F.cross_entropy(logits_intervention_stage2_iter[gts_iter == clsid], gts_iter_cls, reduction='none')
                        loss_anchor_stage2 = F.cross_entropy(logits_anchor_stage2_iter[gts_iter == clsid], gts_iter_cls, reduction='none')
                        relation_mean_stage2 = loss_intervention_stage2.mean() - loss_anchor_stage2.mean()
                        self.class_relations_mean.data[intervention_clsids[batch_idx], clsid] = \
                            relation_mean_stage2 * momentum + self.class_relations_mean.data[intervention_clsids[batch_idx], clsid] * (1 - momentum)
                        if loss_anchor_stage2.shape[0] > 1:
                            relation_var_stage2 = loss_intervention_stage2.var(unbiased=False) - loss_anchor_stage2.var(unbiased=False)
                            self.class_relations_var.data[intervention_clsids[batch_idx], clsid] = \
                                relation_var_stage2 * momentum + self.class_relations_var.data[intervention_clsids[batch_idx], clsid] * (1 - momentum)
                # ----syn
                if dist.is_available() and dist.is_initialized():
                    syn_list = ['class_relations_mean', 'class_relations_var', 'selected_classes_counter']
                    for syn in syn_list:
                        attr = getattr(self, syn).data.clone()
                        dist.all_reduce(attr.div_(dist.get_world_size()))
                        setattr(self, syn, nn.Parameter(attr, requires_grad=False))
            # --update dl_cls_representations
            momentum = self.cfg['head']['dlclsreps_momentum']
            self.updatedlclsreps(feats, data_meta.gettargets()['seg_targets'], momentum, img_size)
            # --calculate losses
            predictions = self.customizepredsandlosses(
                seg_logits=preds_stage2, targets=data_meta.gettargets(), backbone_outputs=backbone_outputs, losses_cfg=self.cfg['losses'], img_size=img_size, auto_calc_loss=False,
            )
            preds_stage2 = predictions.pop('loss_cls')
            preds_stage1 = F.interpolate(preds_stage1, size=img_size, mode='bilinear', align_corners=self.align_corners)
            predictions.update({'loss_cls_stage1': preds_stage1, 'loss_cls_stage2': preds_stage2})
            loss, losses_log_dict = self.calculatelosses(
                predictions=predictions, targets=data_meta.gettargets(), losses_cfg=self.cfg['losses'],
            )
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict) if self.mode == 'TRAIN' else SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict, seg_logits=preds_stage2)
        else:
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, seg_logits=preds_stage2)
        return ssseg_outputs
    '''insert dl_cls_representations into feats'''
    def insertdlrepresentations(self, feats, logits):
        # dl_cls_representations: (num_classes, C)
        dl_cls_representations = self.dl_cls_representations.data.type_as(feats).clone()
        # feats: (batch_size, H, W, C)
        feats = feats.permute(0, 2, 3, 1).contiguous()
        # logits: (batch_size, H, W, num_classes)
        logits = logits.permute(0, 2, 3, 1).contiguous()
        # logits_argmax: (batch_size, H, W)
        logits_argmax = logits.argmax(-1)
        # start to insert
        feats_withdl = torch.zeros(feats.shape[0], feats.shape[1], feats.shape[2], feats.shape[3] * 2).type_as(feats)
        for cls_id in range(self.cfg['num_classes']):
            mask = (logits_argmax == cls_id)
            if mask.sum() == 0: continue
            feats_withdl[mask] = torch.cat([feats[mask], dl_cls_representations[cls_id].unsqueeze(0).expand_as(feats[mask])], dim=1)
        feats_withdl = feats_withdl.permute(0, 3, 1, 2).contiguous()
        # return
        return feats_withdl
    '''obtain intervention-driven context'''
    def obtainidcontext(self, context, logits, class_relations, intervention_clsids=None, remove_negative_cls_relation=True):
        # obtain intervention-driven contextual information
        batch_size, num_channels, context_h, context_w = context.size()
        valid_clsids_batch, id_context_batch = [], torch.zeros_like(context)
        class_relations = class_relations.data.type_as(context).clone()
        for batch_idx in range(batch_size):
            # --context: (num_existing_classes, C), selected_class_relations: (num_classes, num_existing_classes)
            cls_contexts, selected_class_relations = [], []
            # --context_iter: (C, H, W), logits_iter: (num_classes, H, W)
            context_iter, logits_iter = context[batch_idx], logits[batch_idx]
            # --context_iter: (C, H*W), logits_iter: (num_classes, H*W)
            context_iter, logits_iter = context_iter.reshape(num_channels, -1), logits_iter.reshape(self.cfg['num_classes'], -1)
            # --context_iter: (H*W, C)
            context_iter = context_iter.permute(1, 0).contiguous()
            # --logits_iter_argmax: (H*W,)
            logits_iter_argmax = logits_iter.argmax(0)
            valid_clsids = []
            for cls_id in range(self.cfg['num_classes']):
                # --remove intervention clsids
                if intervention_clsids is not None:
                    if cls_id == intervention_clsids[batch_idx]:
                        continue
                # --mask: (H*W,)
                mask = (logits_iter_argmax == cls_id)
                if mask.sum() == 0: continue
                # --context_iter_cls: (N, C)
                context_iter_cls = context_iter[mask]
                # --weight: (N,)
                logits_iter_cls = logits_iter[cls_id, :][mask]
                weight = F.softmax(logits_iter_cls, dim=0)
                # --context_iter_cls: (N, C)
                context_iter_cls = context_iter_cls * weight.unsqueeze(-1)
                # --context_iter_cls: (C,)
                context_iter_cls = context_iter_cls.sum(0)
                # --append
                valid_clsids.append(cls_id)
                cls_contexts.append(context_iter_cls)
                selected_class_relations.append(class_relations[:, cls_id].unsqueeze(1))
            if len(cls_contexts) != 0:
                valid_clsids_batch.append(valid_clsids)
                cls_contexts = torch.stack(cls_contexts)
                selected_class_relations = torch.cat(selected_class_relations, dim=1)
                if remove_negative_cls_relation:
                    selected_class_relations[selected_class_relations <= 0] = -1e16
                selected_class_relations = F.softmax(selected_class_relations, dim=1)
                selected_class_relations_tmp = []
                for cls_id in valid_clsids:
                    selected_class_relations_tmp.append(selected_class_relations[cls_id, :])
                selected_class_relations = torch.stack(selected_class_relations_tmp)
                # --id_context_tmp: (num_existing_classes, C)
                id_context_tmp = torch.matmul(selected_class_relations, cls_contexts)
                # --id_context: (H*W, C)
                id_context = torch.zeros(context_h * context_w, num_channels).type_as(context)
                # --insert
                for idx, cls_id in enumerate(valid_clsids):
                    mask = (logits_iter_argmax == cls_id)
                    assert mask.sum() > 0, 'mask assert error, bug exists'
                    id_context[mask] = id_context_tmp[idx]
                # --id_context: (C, H*W)
                id_context = id_context.permute(1, 0).contiguous()
                # --id_context: (C, H, W)
                id_context = id_context.reshape(num_channels, context_h, context_w)
                # --append
                id_context_batch[batch_idx] = id_context
        # return
        return id_context_batch, valid_clsids_batch
    '''update dl_cls_representations'''
    def updatedlclsreps(self, feats, gts, momentum, img_size):
        with torch.no_grad():
            # feats: (B, H, W, C)
            feats = F.interpolate(feats, size=img_size, mode='bilinear', align_corners=self.align_corners)
            feats = feats.permute(0, 2, 3, 1).contiguous()
            # iter clsids
            unique_cls_ids = gts.unique()
            for cls_id in unique_cls_ids:
                cls_id = int(cls_id.item())
                if cls_id == self.cfg['head']['ignore_index']: continue
                # --feats_cls: (C,)
                feats_cls = feats[gts == cls_id].mean(0)
                # --update
                self.dl_cls_representations.data[cls_id, :] = feats_cls * momentum + self.dl_cls_representations[cls_id, :].clone() * (1 - momentum)
            # sync
            if dist.is_available() and dist.is_initialized():
                dl_cls_representations = self.dl_cls_representations.data.clone()
                dist.all_reduce(dl_cls_representations.div_(dist.get_world_size()))
                self.dl_cls_representations = nn.Parameter(dl_cls_representations, requires_grad=False)