'''
Function:
    Base segmentor for all supported segmentors
Author:
    Zhenchao Jin
'''
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from ...losses import BuildLoss
from ...backbones import BuildBackbone, BuildActivation, BuildNormalization, NormalizationBuilder


'''BaseSegmentor'''
class BaseSegmentor(nn.Module):
    def __init__(self, cfg, mode):
        super(BaseSegmentor, self).__init__()
        self.cfg = cfg
        self.mode = mode
        assert self.mode in ['TRAIN', 'TEST']
        # parse align_corners, normalization layer and activation layer cfg
        self.align_corners, self.norm_cfg, self.act_cfg = cfg['align_corners'], cfg['norm_cfg'], cfg['act_cfg']
        # build backbone
        backbone_cfg = copy.deepcopy(cfg['backbone'])
        if 'norm_cfg' not in backbone_cfg:
            backbone_cfg.update({'norm_cfg': copy.deepcopy(self.norm_cfg)})
        self.backbone_net = BuildBackbone(backbone_cfg)
    '''forward'''
    def forward(self, x, targets=None):
        raise NotImplementedError('not to be implemented')
    '''forward when mode = `TRAIN`'''
    def forwardtrain(self, predictions, targets, backbone_outputs, losses_cfg, img_size, auto_calc_loss=True, map_preds_to_tgts_dict=None):
        predictions = F.interpolate(predictions, size=img_size, mode='bilinear', align_corners=self.align_corners)
        outputs_dict = {'loss_cls': predictions}
        if hasattr(self, 'auxiliary_decoder'):
            backbone_outputs = backbone_outputs[:-1]
            if isinstance(self.auxiliary_decoder, nn.ModuleList):
                assert len(backbone_outputs) >= len(self.auxiliary_decoder)
                backbone_outputs = backbone_outputs[-len(self.auxiliary_decoder):]
                for idx, (out, dec) in enumerate(zip(backbone_outputs, self.auxiliary_decoder)):
                    predictions_aux = dec(out)
                    predictions_aux = F.interpolate(predictions_aux, size=img_size, mode='bilinear', align_corners=self.align_corners)
                    outputs_dict[f'loss_aux{idx+1}'] = predictions_aux
            else:
                predictions_aux = self.auxiliary_decoder(backbone_outputs[-1])
                predictions_aux = F.interpolate(predictions_aux, size=img_size, mode='bilinear', align_corners=self.align_corners)
                outputs_dict = {'loss_cls': predictions, 'loss_aux': predictions_aux}
        if not auto_calc_loss: return outputs_dict
        return self.calculatelosses(predictions=outputs_dict, targets=targets, losses_cfg=losses_cfg, map_preds_to_tgts_dict=map_preds_to_tgts_dict)
    '''forward when mode = `TEST`'''
    def forwardtest(self):
        raise NotImplementedError('not to be implemented')
    '''transforminputs'''
    def transforminputs(self, x_list, selected_indices=None):
        if selected_indices is None:
            if self.cfg['backbone']['type'] in ['HRNet']:
                selected_indices = (0, 0, 0, 0)
            else:
                selected_indices = (0, 1, 2, 3)
        outs = []
        for idx in selected_indices:
            outs.append(x_list[idx])
        return outs
    '''setauxiliarydecoder'''
    def setauxiliarydecoder(self, auxiliary_cfg):
        norm_cfg, act_cfg, num_classes = self.norm_cfg.copy(), self.act_cfg.copy(), self.cfg['num_classes']
        if auxiliary_cfg is None: return
        if isinstance(auxiliary_cfg, dict):
            auxiliary_cfg = [auxiliary_cfg]
        self.auxiliary_decoder = nn.ModuleList()
        for aux_cfg in auxiliary_cfg:
            num_convs = aux_cfg.get('num_convs', 1)
            dec = []
            for idx in range(num_convs):
                if idx == 0:
                    dec += [nn.Conv2d(aux_cfg['in_channels'], aux_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),]
                else:
                    dec += [nn.Conv2d(aux_cfg['out_channels'], aux_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),]
                dec += [
                    BuildNormalization(placeholder=aux_cfg['out_channels'], norm_cfg=norm_cfg),
                    BuildActivation(act_cfg)
                ]
                if 'upsample' in aux_cfg:
                    dec += [nn.Upsample(**aux_cfg['upsample'])]
            dec.append(nn.Dropout2d(aux_cfg['dropout']))
            if num_convs > 0:
                dec.append(nn.Conv2d(aux_cfg['out_channels'], num_classes, kernel_size=1, stride=1, padding=0))
            else:
                dec.append(nn.Conv2d(aux_cfg['in_channels'], num_classes, kernel_size=1, stride=1, padding=0))
            dec = nn.Sequential(*dec)
            self.auxiliary_decoder.append(dec)
        if len(self.auxiliary_decoder) == 1:
            self.auxiliary_decoder = self.auxiliary_decoder[0]
    '''freezenormalization'''
    def freezenormalization(self, norm_list=None):
        if norm_list is None:
            norm_list=(nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
        for module in self.modules():
            if NormalizationBuilder.isnorm(module, norm_list):
                module.eval()
                for p in module.parameters():
                    p.requires_grad = False
    '''calculatelosses'''
    def calculatelosses(self, predictions, targets, losses_cfg, map_preds_to_tgts_dict=None):
        assert len(predictions) == len(losses_cfg), 'length of losses_cfg should be equal to the one of predictions'
        # calculate loss according to losses_cfg
        losses_log_dict = {}
        for loss_name, loss_cfg in losses_cfg.items():
            if map_preds_to_tgts_dict is None:
                losses_log_dict[loss_name] = self.calculateloss(
                    prediction=predictions[loss_name], target=targets['seg_target'], loss_cfg=loss_cfg,
                )
            else:
                losses_log_dict[loss_name] = self.calculateloss(
                    prediction=predictions[loss_name], target=targets[map_preds_to_tgts_dict[loss_name]], loss_cfg=loss_cfg,
                )
        # summarize and convert losses_log_dict
        loss = 0
        for loss_key, loss_value in losses_log_dict.items():
            loss_value = loss_value.mean()
            loss = loss + loss_value
            loss_value = loss_value.data.clone()
            dist.all_reduce(loss_value.div_(dist.get_world_size()))
            losses_log_dict[loss_key] = loss_value.item()
        losses_log_dict.update({'loss_total': sum(losses_log_dict.values())})
        # return the loss and losses_log_dict
        return loss, losses_log_dict
    '''calculateloss'''
    def calculateloss(self, prediction, target, loss_cfg):
        assert isinstance(loss_cfg, (dict, list))
        # calculate the loss, dict means single-type loss and list represents multiple-type losses
        if isinstance(loss_cfg, dict):
            loss = BuildLoss(loss_cfg)(prediction, target)
        else:
            loss = 0
            for l_cfg in loss_cfg:
                loss = loss + BuildLoss(l_cfg)(prediction, target)
        # return the loss
        return loss