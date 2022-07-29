'''
Function:
    Define the params constructor
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn


'''DefaultParamsConstructor'''
class DefaultParamsConstructor():
    def __init__(self, params_rules={}, filter_params=False, optimizer_cfg=None):
        self.params_rules = params_rules
        self.filter_params = filter_params
        self.optimizer_cfg = optimizer_cfg
    '''call'''
    def __call__(self, model):
        params_rules, filter_params, optimizer_cfg = self.params_rules, self.filter_params, self.optimizer_cfg
        if params_rules:
            params, all_layers = [], model.alllayers()
            assert 'others' not in all_layers, 'potential bug in model.alllayers'
            for key, value in params_rules.items():
                if not isinstance(value, tuple): value = (value, value)
                if key == 'others': continue
                params.append({
                    'params': all_layers[key].parameters() if not filter_params else filter(lambda p: p.requires_grad, all_layers[key].parameters()), 
                    'lr': optimizer_cfg['lr'] * value[0], 
                    'name': key,
                    'weight_decay': optimizer_cfg['weight_decay'] * value[1],
                })
            others = []
            for key, layer in all_layers.items():
                if key not in params_rules: others.append(layer)
            others = nn.Sequential(*others)
            value = (params_rules['others'], params_rules['others']) if not isinstance(params_rules['others'], tuple) else params_rules['others']
            params.append({
                'params': others.parameters() if not filter_params else filter(lambda p: p.requires_grad, others.parameters()), 
                'lr': optimizer_cfg['lr'] * value[0], 
                'name': 'others',
                'weight_decay': optimizer_cfg['weight_decay'] * value[1],
            })
        else:
            params = model.parameters() if not filter_params else filter(lambda p: p.requires_grad, model.parameters())
        return params


'''LayerDecayParamsConstructor'''
class LayerDecayParamsConstructor():
    def __init__(self, params_rules={}, filter_params=True, optimizer_cfg=None):
        self.params_rules = params_rules
        self.filter_params = filter_params
        self.optimizer_cfg = optimizer_cfg
        setattr(self, 'filter_params', True)
    '''call'''
    def __call__(self, model):
        # parse config
        params_rules, filter_params, optimizer_cfg = self.params_rules, self.filter_params, self.optimizer_cfg
        num_layers = params_rules['num_layers'] + 2
        decay_rate = params_rules['decay_rate']
        decay_type = params_rules['decay_type']
        weight_decay = optimizer_cfg['weight_decay']
        base_lr = optimizer_cfg['lr']
        # iter params
        params, parameter_groups = [], {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name.endswith('.bias') or name in ('pos_embed', 'cls_token'):
                group_name = 'no_decay'
                this_weight_decay = 0.
            else:
                group_name = 'decay'
                this_weight_decay = weight_decay
            if 'layer_wise' in decay_type:
                if 'ConvNeXt' in model.backbone_net.__class__.__name__:
                    layer_id = self.getlayeridforconvnext(name, params_rules['num_layers'])
                elif 'BEiT' in model.backbone_net.__class__.__name__ or 'MAE' in model.backbone_net.__class__.__name__:
                    layer_id = self.getlayeridforvit(name, num_layers)
                else:
                    raise NotImplementedError('not to be implemented')
            elif decay_type == 'stage_wise':
                if 'ConvNeXt' in model.backbone_net.__class__.__name__:
                    layer_id = self.getstageidforconvnext(name, num_layers)
                else:
                    raise NotImplementedError('not to be implemented')
            group_name = f'layer_{layer_id}_{group_name}'
            if group_name not in parameter_groups:
                scale = decay_rate**(num_layers - layer_id - 1)
                parameter_groups[group_name] = {
                    'weight_decay': this_weight_decay,
                    'params': [],
                    'param_names': [],
                    'lr_scale': scale,
                    'group_name': group_name,
                    'lr': scale * base_lr,
                }
            parameter_groups[group_name]['params'].append(param)
            parameter_groups[group_name]['param_names'].append(name)
        params.extend(parameter_groups.values())
        # return
        return params
    '''getlayeridforconvnext'''
    def getlayeridforconvnext(self, var_name, max_layer_id):
        if var_name in ('backbone_net.cls_token', 'backbone_net.mask_token', 'backbone_net.pos_embed'):
            return 0
        elif var_name.startswith('backbone_net.downsample_layers'):
            stage_id = int(var_name.split('.')[2])
            if stage_id == 0:
                layer_id = 0
            elif stage_id == 1:
                layer_id = 2
            elif stage_id == 2:
                layer_id = 3
            elif stage_id == 3:
                layer_id = max_layer_id
            return layer_id
        elif var_name.startswith('backbone_net.stages'):
            stage_id = int(var_name.split('.')[2])
            block_id = int(var_name.split('.')[3])
            if stage_id == 0:
                layer_id = 1
            elif stage_id == 1:
                layer_id = 2
            elif stage_id == 2:
                layer_id = 3 + block_id // 3
            elif stage_id == 3:
                layer_id = max_layer_id
            return layer_id
        else:
            return max_layer_id + 1
    '''getstageidforconvnext'''
    def getstageidforconvnext(self, var_name, max_stage_id):
        if var_name in ('backbone_net.cls_token', 'backbone_net.mask_token', 'backbone_net.pos_embed'):
            return 0
        elif var_name.startswith('backbone_net.downsample_layers'):
            return 0
        elif var_name.startswith('backbone_net.stages'):
            stage_id = int(var_name.split('.')[2])
            return stage_id + 1
        else:
            return max_stage_id - 1
    '''getlayeridforvit'''
    def getlayeridforvit(self, var_name, max_layer_id):
        if var_name in ('backbone_net.cls_token', 'backbone_net.mask_token', 'backbone_net.pos_embed'):
            return 0
        elif var_name.startswith('backbone_net.patch_embed'):
            return 0
        elif var_name.startswith('backbone_net.layers'):
            layer_id = int(var_name.split('.')[2])
            return layer_id + 1
        else:
            return max_layer_id - 1