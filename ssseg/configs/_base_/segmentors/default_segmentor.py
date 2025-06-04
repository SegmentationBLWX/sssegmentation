'''default_segmentor'''
import copy
from typing import Any, Dict
from dataclasses import dataclass, field
from collections.abc import MutableMapping


'''SegmentorConfig'''
@dataclass
class SegmentorConfig(MutableMapping):
    type: str
    num_classes: int
    benchmark: bool
    align_corners: bool
    work_dir: str
    eval_interval_epochs: int
    save_interval_epochs: int
    logger_handle_cfg: Dict[str, Any]
    training_logging_manager_cfg: Dict[str, Any]
    norm_cfg: Dict[str, Any]
    act_cfg: Dict[str, Any]
    backbone: Dict[str, Any]
    head: Dict[str, Any]
    auxiliary: Dict[str, Any]
    losses: Dict[str, Any]
    inference: Dict[str, Any]
    scheduler: Dict[str, Any]
    dataset: Any = None
    dataloader: Any = None
    _extra: Dict[str, Any] = field(default_factory=dict, repr=False)
    def __init__(self, **kwargs):
        keys = {
            'type', 'num_classes', 'benchmark', 'align_corners', 'work_dir', 'eval_interval_epochs', 'save_interval_epochs', 'logger_handle_cfg', 'training_logging_manager_cfg',
            'norm_cfg', 'act_cfg', 'backbone', 'head', 'auxiliary', 'losses', 'inference', 'scheduler', 'dataset', 'dataloader'
        }
        for key in keys:
            setattr(self, key, kwargs.pop(key, None))
        self._extra = kwargs
    '''getitem'''
    def __getitem__(self, key):
        if key in self.__dict__:
            return getattr(self, key)
        return self._extra[key]
    '''setitem'''
    def __setitem__(self, key, value):
        if key in self.__dict__:
            setattr(self, key, value)
        else:
            self._extra[key] = value
    '''delitem'''
    def __delitem__(self, key):
        if key in self.__dict__:
            delattr(self, key)
        elif key in self._extra:
            del self._extra[key]
        else:
            raise KeyError(key)
    '''iter'''
    def __iter__(self):
        core = {k: v for k, v in self.__dict__.items() if k != '_extra'}
        return iter({**core, **self._extra})
    '''len'''
    def __len__(self):
        return len(self.__dict__) - 1 + len(self._extra)
    '''repr'''
    def __repr__(self):
        core = {k: v for k, v in self.__dict__.items() if k != '_extra'}
        merged = {**core, **self._extra}
        return f"SegmentorConfig({merged})"
    '''copy'''
    def copy(self):
        return copy.deepcopy(self)