'''
Function:
    Implementation of ParamsConstructors
Author:
    Zhenchao Jin
'''
import copy
from ...utils import BaseModuleBuilder
from ..backbones import NormalizationBuilder


'''DefaultParamsConstructor'''
class DefaultParamsConstructor():
    def __init__(self, params_rules={}, filter_params=False, optimizer_cfg=None):
        self.params_rules = params_rules
        self.filter_params = filter_params
        self.optimizer_cfg = optimizer_cfg
    '''call'''
    def __call__(self, model):
        # fetch attributes
        params_rules, filter_params, optimizer_cfg = self.params_rules, self.filter_params, self.optimizer_cfg
        # without specific parameter rules
        if not params_rules:
            params = model.parameters() if not filter_params else filter(lambda p: p.requires_grad, model.parameters())
            return params
        # with specific parameter rules
        params = []
        self.groupparams(model, params_rules, filter_params, optimizer_cfg, params)
        return params
    '''groupparams'''
    def groupparams(self, model, params_rules, filter_params, optimizer_cfg, params, prefix=''):
        # fetch base_setting
        optimizer_cfg = copy.deepcopy(optimizer_cfg)
        if 'base_setting' in optimizer_cfg:
            base_setting = optimizer_cfg.pop('base_setting')
        else:
            base_setting = {
                'bias_lr_multiplier': 1.0, 'bias_wd_multiplier': 1.0, 'norm_wd_multiplier': 1.0,
                'lr_multiplier': 1.0, 'wd_multiplier': 1.0
            }
        # iter to group current parameters
        sorted_rule_keys = sorted(sorted(params_rules.keys()), key=len, reverse=True)
        for name, param in model.named_parameters(recurse=False):
            param_group = {'params': [param]}
            # --if `parameter requires gradient` is False
            if not param.requires_grad:
                if not filter_params:
                    params.append(param_group)
                continue
            # --find parameters with specific rules
            set_base_setting = True
            for rule_key in sorted_rule_keys:
                if rule_key not in f'{prefix}.{name}': continue
                set_base_setting = False
                param_group['lr'] = params_rules[rule_key].get('lr_multiplier', 1.0) * optimizer_cfg['lr']
                param_group['name'] = f'{prefix}.{name}' if prefix else name
                param_group['rule_key'] = rule_key
                if 'weight_decay' in optimizer_cfg:
                    param_group['weight_decay'] = params_rules[rule_key].get('wd_multiplier', 1.0) * optimizer_cfg['weight_decay']
                for k, v in params_rules[rule_key].items():
                    assert k not in param_group, 'construct param_group error'
                    param_group[k] = v
                params.append(param_group)
                break
            if not set_base_setting: continue
            # --set base setting
            param_group['lr'] = optimizer_cfg['lr']
            param_group['name'] = f'{prefix}.{name}' if prefix else name
            param_group['rule_key'] = 'base_setting'
            if name == 'bias' and (not NormalizationBuilder.isnorm(model)):
                param_group['lr'] = param_group['lr'] * base_setting.get('bias_lr_multiplier', 1.0)
                param_group['lr_multiplier'] = base_setting.get('bias_lr_multiplier', 1.0)
            else:
                param_group['lr'] = param_group['lr'] * base_setting.get('lr_multiplier', 1.0)
                param_group['lr_multiplier'] = base_setting.get('lr_multiplier', 1.0)
            if 'weight_decay' in optimizer_cfg:
                param_group['weight_decay'] = optimizer_cfg['weight_decay']
                if NormalizationBuilder.isnorm(model):
                    param_group['weight_decay'] = param_group['weight_decay'] * base_setting.get('norm_wd_multiplier', 1.0)
                elif name == 'bias':
                    param_group['weight_decay'] = param_group['weight_decay'] * base_setting.get('bias_wd_multiplier', 1.0)
                else:
                    param_group['weight_decay'] = param_group['weight_decay'] * base_setting.get('wd_multiplier', 1.0)
            params.append(param_group)
        # iter to group children parameters
        for child_name, child_model in model.named_children():
            if prefix:
                child_prefix = f'{prefix}.{child_name}'
            else:
                child_prefix = child_name
            self.groupparams(child_model, params_rules, filter_params, optimizer_cfg, params, prefix=child_prefix)
    '''isin'''
    def isin(self, param_group, param_group_list):
        param = set(param_group['params'])
        param_set = set()
        for group in param_group_list:
            param_set.update(set(group['params']))
        return not param.isdisjoint(param_set)


'''LearningRateDecayParamsConstructor'''
class LearningRateDecayParamsConstructor(DefaultParamsConstructor):
    def __init__(self, params_rules={}, filter_params=True, optimizer_cfg=None):
        # force filter_params as True
        filter_params = True
        super(LearningRateDecayParamsConstructor, self).__init__(
            params_rules=params_rules, filter_params=filter_params, optimizer_cfg=optimizer_cfg,
        )
    '''groupparams'''
    def groupparams(self, model, params_rules, filter_params, optimizer_cfg, params, prefix=''):
        # parse params_rules
        num_layers = params_rules['num_layers'] + 2
        decay_rate = params_rules['decay_rate']
        decay_type = params_rules['decay_type']
        base_lr = optimizer_cfg['lr']
        weight_decay = optimizer_cfg['weight_decay']
        # iter to group parameters
        parameter_groups = {}
        for name, param in model.named_parameters():
            param_group = {'params': [param]}
            # --if `parameter requires gradient` is False
            if not param.requires_grad:
                if not filter_params:
                    params.append(param_group)
                continue
            # --find parameters with specific weight_decay rules
            if len(param.shape) == 1 or name.endswith('.bias') or name in ('pos_embed', 'cls_token'):
                group_name = 'no_decay'
                this_weight_decay = 0.
            else:
                group_name = 'decay'
                this_weight_decay = weight_decay
            # --set layer_id
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
            # --set group_name and thus append to parameter_groups
            group_name = f'layer_{layer_id}_{group_name}'
            if group_name not in parameter_groups:
                scale = decay_rate**(num_layers - layer_id - 1)
                parameter_groups[group_name] = {
                    'weight_decay': this_weight_decay, 'params': [], 'param_names': [], 'lr_multiplier': scale, 'group_name': group_name, 'lr': scale * base_lr,
                }
            parameter_groups[group_name]['params'].append(param)
            parameter_groups[group_name]['param_names'].append(name)
        params.extend(parameter_groups.values())
    '''getlayeridforconvnext'''
    def getlayeridforconvnext(self, var_name, max_layer_id):
        if var_name in ('backbone_net.cls_token', 'backbone_net.mask_token', 'backbone_net.pos_embed'):
            return 0
        elif var_name.startswith('backbone_net.downsample_layers'):
            stage_id = int(var_name.split('.')[2])
            if stage_id == 0: layer_id = 0
            elif stage_id == 1: layer_id = 2
            elif stage_id == 2: layer_id = 3
            elif stage_id == 3: layer_id = max_layer_id
            return layer_id
        elif var_name.startswith('backbone_net.stages'):
            stage_id = int(var_name.split('.')[2])
            block_id = int(var_name.split('.')[3])
            if stage_id == 0: layer_id = 1
            elif stage_id == 1: layer_id = 2
            elif stage_id == 2: layer_id = 3 + block_id // 3
            elif stage_id == 3: layer_id = max_layer_id
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


'''ParamsConstructorBuilder'''
class ParamsConstructorBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'DefaultParamsConstructor': DefaultParamsConstructor, 'LearningRateDecayParamsConstructor': LearningRateDecayParamsConstructor,
    }
    '''build'''
    def build(self, params_rules={}, filter_params=False, optimizer_cfg={}):
        constructor_type = params_rules.pop('type', 'DefaultParamsConstructor')
        module_cfg = {
            'params_rules': params_rules, 'filter_params': filter_params, 'optimizer_cfg': optimizer_cfg, 'type': constructor_type
        }
        return super().build(module_cfg)


'''BuildParamsConstructor'''
BuildParamsConstructor = ParamsConstructorBuilder().build