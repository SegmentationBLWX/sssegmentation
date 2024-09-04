'''
Function:
    Implementation of DataStructure
Author:
    Zhenchao Jin
'''
import torch
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
    '''gettargets'''
    def gettargets(self):
        targets = {}
        if self.seg_targets is not None:
            targets.update({'seg_targets': self.seg_targets})
        if self.edge_targets is not None:
            targets.update({'edge_targets': self.edge_targets})
        return targets