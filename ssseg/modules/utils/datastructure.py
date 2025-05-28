'''
Function:
    Implementation of DataStructure
Author:
    Zhenchao Jin
'''
import copy
import torch
from typing import List
from typing import Optional
from dataclasses import dataclass, field


'''SSSegOutputStructure'''
@dataclass
class SSSegOutputStructure:
    mode: Optional[str] = field(default='TRAIN')
    auto_validate: Optional[bool] = field(default=True)
    loss: Optional[torch.Tensor] = field(default=None)
    seg_logits: Optional[torch.Tensor] = field(default=None)
    losses_log_dict: Optional[dict] = field(default=None)
    '''postinit'''
    def __post_init__(self):
        self.validate()
    '''getitem'''
    def __getitem__(self, idx) -> "SSSegOutputStructure":
        all_fields = self.getsetvariables()
        init_field_names = self.__dataclass_fields__.keys()
        init_fields, extra_fields = {}, {}
        for k, v in all_fields.items():
            if isinstance(v, torch.Tensor):
                sliced_v = v[idx]
            else:
                raise TypeError(f"unsupported field type for indexing: {k} ({type(v)})")
            if k in init_field_names:
                init_fields[k] = sliced_v
            else:
                extra_fields[k] = sliced_v
        instance = SSSegOutputStructure(mode=self.mode, auto_validate=self.auto_validate, **init_fields)
        for k, v in extra_fields.items():
            instance.addfield(k, v)
        return instance
    '''addfield'''
    def addfield(self, field_name, field_value):
        assert not hasattr(self, field_name), f'field_name {field_name} already exits, call .setvariable if you want set value for it'
        setattr(self, field_name, field_value)
    '''delfield'''
    def delfield(self, field_name):
        assert hasattr(self, field_name), f'field_name {field_name} does not exits'
        delattr(self, field_name)
    '''validate'''
    def validate(self):
        if not self.auto_validate: return
        assert self.mode in ['TRAIN', 'TEST', 'TRAIN_DEVELOP']
        if self.mode == 'TRAIN':
            if self.loss is None or self.losses_log_dict is None:
                raise ValueError('loss and losses_log_dict should be set if mode is TRAIN')
            if self.seg_logits is not None:
                raise ValueError('seg_logits should not be set if mode is TRAIN')
        elif self.mode == 'TEST':
            if self.loss is not None or self.losses_log_dict is not None:
                raise ValueError('loss and losses_log_dict should not be set if mode is TEST')
            if self.seg_logits is None:
                raise ValueError('seg_logits should be set if mode is TEST')
        elif self.mode == 'TRAIN_DEVELOP':
             if self.loss is None or self.losses_log_dict is None or self.seg_logits is None:
                 raise ValueError('loss, losses_log_dict and seg_logits should be set if mode is TRAIN_DEVELOP')
    '''setvariable'''
    def setvariable(self, field_name, field_value):
        if not hasattr(self, field_name):
            raise AttributeError(f"field_name {field_name} does not exist")
        setattr(self, field_name, field_value)
        self.validate()
    '''getsetvariables'''
    def getsetvariables(self):
        return {k: v for k, v in self.__dict__.items() if v is not None and k not in ['mode', 'auto_validate']}
    '''stack'''
    @staticmethod
    def stack(instances: List["SSSegOutputStructure"], dim: int = 0) -> "SSSegOutputStructure":
        assert all(isinstance(x, SSSegOutputStructure) for x in instances), "all inputs should be SSSegOutputStructure"
        assert len(instances) > 0, "no instances to stack"
        ref = instances[0]
        stacked = copy.deepcopy(ref)
        for field_name, _ in ref.getsetvariables().items():
            all_vals = [getattr(x, field_name) for x in instances]
            if all(isinstance(v, torch.Tensor) for v in all_vals):
                stacked.setvariable(field_name, torch.stack(all_vals, dim=dim))
            else:
                raise TypeError(f"cannot concatenate non-Tensor field: {field_name}")
        return stacked


'''SSSegInputStructure'''
@dataclass
class SSSegInputStructure:
    mode: Optional[str] = field(default='TRAIN')
    auto_validate: Optional[bool] = field(default=True)
    images: Optional[torch.Tensor] = field(default=None)
    seg_targets: Optional[torch.Tensor] = field(default=None)
    edge_targets: Optional[torch.Tensor] = field(default=None)
    img2aug_pos_mapper: Optional[torch.Tensor] = field(default=None)
    '''postinit'''
    def __post_init__(self):
        self.validate()
    '''getitem'''
    def __getitem__(self, idx) -> "SSSegInputStructure":
        all_fields = self.getsetvariables()
        init_field_names = self.__dataclass_fields__.keys()
        init_fields, extra_fields = {}, {}
        for k, v in all_fields.items():
            if isinstance(v, torch.Tensor):
                sliced_v = v[idx]
            else:
                raise TypeError(f"unsupported field type for indexing: {k} ({type(v)})")
            if k in init_field_names:
                init_fields[k] = sliced_v
            else:
                extra_fields[k] = sliced_v
        instance = SSSegInputStructure(mode=self.mode, auto_validate=self.auto_validate, **init_fields)
        for k, v in extra_fields.items():
            instance.addfield(k, v)
        return instance
    '''addfield'''
    def addfield(self, field_name, field_value):
        assert not hasattr(self, field_name), f'field_name {field_name} already exits, call .setvariable if you want set value for it'
        setattr(self, field_name, field_value)
    '''delfield'''
    def delfield(self, field_name):
        assert hasattr(self, field_name), f'field_name {field_name} does not exits'
        delattr(self, field_name)
    '''validate'''
    def validate(self):
        if not self.auto_validate: return
        assert self.mode in ['TRAIN', 'TEST', 'TRAIN_DEVELOP']
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            if self.images is None or (self.seg_targets is None and self.edge_targets is None):
                raise ValueError('images, (seg_targets or edge_targets) should be set if mode is TRAIN and TRAIN_DEVELOP')
        elif self.mode == 'TEST':
            if self.images is None:
                raise ValueError('images should be set if mode is TEST')
    '''setvariable'''
    def setvariable(self, field_name, field_value):
        if not hasattr(self, field_name):
            raise AttributeError(f"field_name {field_name} does not exist")
        setattr(self, field_name, field_value)
        self.validate()
    '''getsetvariables'''
    def getsetvariables(self):
        return {k: v for k, v in self.__dict__.items() if v is not None and k not in ['mode', 'auto_validate']}
    '''getannotations'''
    def getannotations(self):
        annotations = {}
        if self.seg_targets is not None:
            annotations.update({'seg_targets': self.seg_targets})
        if self.edge_targets is not None:
            annotations.update({'edge_targets': self.edge_targets})
        return annotations
    '''stack'''
    @staticmethod
    def stack(instances: List["SSSegInputStructure"], dim: int = 0) -> "SSSegInputStructure":
        assert all(isinstance(x, SSSegInputStructure) for x in instances), "all inputs should be SSSegInputStructure"
        assert len(instances) > 0, "no instances to stack"
        ref = instances[0]
        stacked = copy.deepcopy(ref)
        for field_name, _ in ref.getsetvariables().items():
            all_vals = [getattr(x, field_name) for x in instances]
            if all(isinstance(v, torch.Tensor) for v in all_vals):
                stacked.setvariable(field_name, torch.stack(all_vals, dim=dim))
            else:
                raise TypeError(f"cannot concatenate non-Tensor field: {field_name}")
        return stacked