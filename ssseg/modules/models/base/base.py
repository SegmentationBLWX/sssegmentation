'''
Function:
    Base model for all supported models
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
import torch.distributed as dist
from ...backbones import *
from ...losses import BuildLoss


'''base model'''
class BaseModel(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(BaseModel, self).__init__()
        self.cfg = cfg
        self.mode = kwargs.get('mode')
        assert self.mode in ['TRAIN', 'TEST']
        # parse align_corners, normalization layer and activation layer cfg
        self.align_corners, self.norm_cfg, self.act_cfg = cfg['align_corners'], cfg['norm_cfg'], cfg['act_cfg']
        # build backbone
        backbone_cfg = copy.deepcopy(cfg['backbone'])
        if 'norm_cfg' not in backbone_cfg:
            backbone_cfg.update({'norm_cfg': copy.deepcopy(self.norm_cfg)})
        self.backbone_net = BuildBackbone(backbone_cfg)
    '''forward'''
    def forward(self, x, targets=None, losses_cfg=None):
        raise NotImplementedError('not to be implemented')
    '''transform inputs'''
    def transforminputs(self, x_list, selected_indices=None):
        if selected_indices is None:
            if self.cfg['backbone']['series'] in ['hrnet']:
                selected_indices = (0, 0, 0, 0)
            else:
                selected_indices = (0, 1, 2, 3)
        outs = []
        for idx in selected_indices:
            outs.append(x_list[idx])
        return outs
    '''return all layers with learnable parameters'''
    def alllayers(self):
        raise NotImplementedError('not to be implemented')
    '''freeze normalization'''
    def freezenormalization(self):
        for module in self.modules():
            if type(module) in BuildNormalization(only_get_all_supported=True):
                module.eval()
    '''calculate the losses'''
    def calculatelosses(self, predictions, targets, losses_cfg, targets_keys_dict=None):
        # parse targets
        target_seg = targets['segmentation']
        if 'edge' in targets:
            target_edge = targets['edge']
            num_neg_edge, num_pos_edge = torch.sum(target_edge == 0, dtype=torch.float), torch.sum(target_edge == 1, dtype=torch.float)
            weight_pos_edge, weight_neg_edge = num_neg_edge / (num_pos_edge + num_neg_edge), num_pos_edge / (num_pos_edge + num_neg_edge)
            cls_weight_edge = torch.Tensor([weight_neg_edge, weight_pos_edge]).type_as(target_edge)
        # calculate loss according to losses_cfg
        assert len(predictions) == len(losses_cfg), 'length of losses_cfg should be equal to predictions...'
        losses_log_dict = {}
        for loss_name, loss_cfg in losses_cfg.items():
            if targets_keys_dict is None:
                if 'edge' in loss_name:
                    loss_cfg = copy.deepcopy(loss_cfg)
                    loss_cfg_keys = loss_cfg.keys()
                    for key in loss_cfg_keys: loss_cfg[key]['opts'].update({'weight': cls_weight_edge})
                    losses_log_dict[loss_name] = self.calculateloss(
                        prediction=predictions[loss_name],
                        target=target_edge,
                        loss_cfg=loss_cfg,
                    )
                else:
                    losses_log_dict[loss_name] = self.calculateloss(
                        prediction=predictions[loss_name],
                        target=target_seg,
                        loss_cfg=loss_cfg,
                    )
            else:
                losses_log_dict[loss_name] = self.calculateloss(
                    prediction=predictions[loss_name],
                    target=targets[targets_keys_dict[loss_name]],
                    loss_cfg=loss_cfg,
                )
        loss = 0
        for key, value in losses_log_dict.items():
            value = value.mean()
            loss += value
            losses_log_dict[key] = value
        losses_log_dict.update({'total': loss})
        # convert losses_log_dict
        for key, value in losses_log_dict.items():
            if dist.is_available() and dist.is_initialized():
                value = value.data.clone()
                dist.all_reduce(value.div_(dist.get_world_size()))
                losses_log_dict[key] = value.item()
            else:
                losses_log_dict[key] = torch.Tensor([value.item()]).type_as(loss)
        # return the loss and losses_log_dict
        return loss, losses_log_dict
    '''calculate the loss'''
    def calculateloss(self, prediction, target, loss_cfg):
        # format prediction
        if prediction.dim() == 4:
            prediction_format = prediction.permute((0, 2, 3, 1)).contiguous()
        elif prediction.dim() == 3:
            prediction_format = prediction.permute((0, 2, 1)).contiguous()
        else:
            prediction_format = prediction
        prediction_format = prediction_format.view(-1, prediction_format.size(-1))
        # calculate the loss
        loss = 0
        for key, value in loss_cfg.items():
            if (key in ['binaryceloss']) and hasattr(self, 'onehot'):
                prediction_iter = prediction_format
                target_iter = self.onehot(target, self.cfg['num_classes'])
            elif key in ['diceloss', 'lovaszloss', 'kldivloss']:
                prediction_iter = prediction
                target_iter = target
            else:
                prediction_iter = prediction_format
                target_iter = target.view(-1)
            loss += BuildLoss(key)(
                prediction=prediction_iter, 
                target=target_iter, 
                scale_factor=value['scale_factor'],
                **value['opts']
            )
        # return the loss
        return loss