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
from ...losses import *
from ...backbones import *


'''base model'''
class BaseModel(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(BaseModel, self).__init__()
        self.mode = kwargs.get('mode')
        assert self.mode in ['TRAIN', 'TEST']
        # parse align_corners, normalization layer and activation layer cfg
        self.align_corners, self.normlayer_opts, self.activation_opts = cfg['align_corners'], cfg['normlayer_opts'], cfg['activation_opts']
        # build backbone
        backbone_cfg = copy.deepcopy(cfg['backbone'])
        backbone_cfg.update({'normlayer_opts': copy.deepcopy(self.normlayer_opts)})
        self.backbone_net = BuildBackbone(backbone_cfg)
    '''forward'''
    def forward(self, x, targets=None, losses_cfg=None):
        raise NotImplementedError('not to be implemented')
    '''return all layers with learnable parameters'''
    def alllayers(self):
        raise NotImplementedError('not to be implemented')
    '''freeze normalization layer'''
    def freezenormlayer(self):
        for module in self.modules():
            if type(module) in BuildNormalizationLayer(only_get_all_supported=True):
                module.eval()
    '''calculate the losses'''
    def calculatelosses(self, predictions, targets, losses_cfg):
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
            if 'edge' in loss_name:
                loss_cfg = copy.deepcopy(loss_cfg)
                loss_cfg_keys = loss_cfg.keys()
                for key in loss_cfg_keys: loss_cfg[key]['opts'].update({'cls_weight': cls_weight_edge})
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
        # return the loss and losses_log_dict
        return loss, losses_log_dict
    '''calculate the loss'''
    def calculateloss(self, prediction, target, loss_cfg):
        # define the supported losses
        supported_losses = {
            'celoss': CrossEntropyLoss,
            'sigmoidfocalloss': SigmoidFocalLoss,
        }
        # calculate the loss
        loss = 0
        prediction = prediction.permute((0, 2, 3, 1)).contiguous()
        for key, value in loss_cfg.items():
            assert key in supported_losses, 'unsupport loss type %s...' % key
            loss += supported_losses[key](
                prediction=prediction.view(-1, prediction.size(3)), 
                target=target.view(-1), 
                scale_factor=value['scale_factor'],
                **value['opts']
            )
        # return the loss
        return loss