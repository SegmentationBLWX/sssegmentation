'''
Function:
    Base segmentor for all supported segmentors
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from ...losses import BuildLoss
from ...backbones import BuildBackbone, BuildActivation, BuildNormalization, constructnormcfg


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
    def forward(self, x, targets=None, losses_cfg=None):
        raise NotImplementedError('not to be implemented')
    '''forward when mode = `TRAIN`'''
    def forwardtrain(self, predictions, targets, backbone_outputs, losses_cfg, img_size, compute_loss=True):
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
        if not compute_loss: 
            return outputs_dict
        return self.calculatelosses(
            predictions=outputs_dict, 
            targets=targets, 
            losses_cfg=losses_cfg
        )
    '''forward when mode = `TEST`'''
    def forwardtest(self):
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
        assert len(self.layer_names) == len(set(self.layer_names))
        require_training_layers = {}
        for layer_name in self.layer_names:
            if hasattr(self, layer_name) and layer_name not in ['backbone_net']:
                require_training_layers[layer_name] = getattr(self, layer_name)
            elif hasattr(self, layer_name) and layer_name in ['backbone_net']:
                if hasattr(getattr(self, layer_name), 'nonzerowdlayers'):
                    assert hasattr(getattr(self, layer_name), 'zerowdlayers')
                    tmp_layers = []
                    for key, value in getattr(self, layer_name).zerowdlayers().items():
                        tmp_layers.append(value)
                    require_training_layers.update({f'{layer_name}_zerowd': nn.Sequential(*tmp_layers)})
                    tmp_layers = []
                    for key, value in getattr(self, layer_name).nonzerowdlayers().items():
                        tmp_layers.append(value)
                    require_training_layers.update({f'{layer_name}_nonzerowd': nn.Sequential(*tmp_layers)})
                else:
                    require_training_layers[layer_name] = getattr(self, layer_name)
            elif hasattr(self, layer_name):
                raise NotImplementedError(f'layer name {layer_name} error')
        return require_training_layers
    '''set auxiliary decoder as attribute'''
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
                    BuildNormalization(constructnormcfg(placeholder=aux_cfg['out_channels'], norm_cfg=norm_cfg)),
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
    '''freeze normalization'''
    def freezenormalization(self):
        for module in self.modules():
            if type(module) in BuildNormalization(only_get_all_supported=True):
                module.eval()
    '''calculate the losses'''
    def calculatelosses(self, predictions, targets, losses_cfg, map_preds_to_tgts_dict=None):
        # parse targets
        target_seg = targets['segmentation']
        if 'edge' in targets:
            target_edge = targets['edge']
            num_neg_edge, num_pos_edge = torch.sum(target_edge == 0, dtype=torch.float), torch.sum(target_edge == 1, dtype=torch.float)
            weight_pos_edge, weight_neg_edge = num_neg_edge / (num_pos_edge + num_neg_edge), num_pos_edge / (num_pos_edge + num_neg_edge)
            cls_weight_edge = torch.Tensor([weight_neg_edge, weight_pos_edge]).type_as(target_edge)
        # calculate loss according to losses_cfg
        assert len(predictions) == len(losses_cfg), 'length of losses_cfg should be equal to predictions'
        losses_log_dict = {}
        for loss_name, loss_cfg in losses_cfg.items():
            if 'edge' in loss_name:
                loss_cfg = copy.deepcopy(loss_cfg)
                loss_cfg_keys = loss_cfg.keys()
                for key in loss_cfg_keys:
                    loss_cfg[key].update({'weight': cls_weight_edge})
            if map_preds_to_tgts_dict is None:
                losses_log_dict[loss_name] = self.calculateloss(
                    prediction=predictions[loss_name],
                    target=target_edge if 'edge' in loss_name else target_seg,
                    loss_cfg=loss_cfg,
                )
            else:
                losses_log_dict[loss_name] = self.calculateloss(
                    prediction=predictions[loss_name],
                    target=targets[map_preds_to_tgts_dict[loss_name]],
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
            elif key in ['diceloss', 'lovaszloss', 'kldivloss', 'l1loss', 'cosinesimilarityloss']:
                prediction_iter = prediction
                target_iter = target
            else:
                prediction_iter = prediction_format
                target_iter = target.view(-1)
            loss += BuildLoss(key)(
                prediction=prediction_iter, 
                target=target_iter, 
                **value
            )
        # return the loss
        return loss