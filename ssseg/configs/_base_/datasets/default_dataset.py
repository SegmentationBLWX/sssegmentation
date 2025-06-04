'''default_dataset'''
import copy
from typing import Any, Dict
from dataclasses import dataclass, field
from collections.abc import MutableMapping


'''DatasetConfig'''
@dataclass
class DatasetConfig(MutableMapping):
    type: str
    rootdir: str
    train: Dict[str, Any] = field(default_factory=dict)
    test: Dict[str, Any] = field(default_factory=dict)
    _extra: Dict[str, Any] = field(default_factory=dict, repr=False)
    def __init__(self, type: str, rootdir: str, train=None, test=None, **kwargs):
        self.type = type
        self.rootdir = rootdir
        self.train = train if train is not None else {}
        self.test = test if test is not None else {}
        self._extra = kwargs
    '''getitem'''
    def __getitem__(self, key):
        if key in self.__dict__:
            return getattr(self, key)
        return self._extra[key]
    '''setitem'''
    def __setitem__(self, key, value):
        if key in {'type', 'rootdir', 'train', 'test'}:
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
        return iter({**{k: v for k, v in self.__dict__.items() if k != '_extra'}, **self._extra})
    '''len'''
    def __len__(self):
        return len(self.__dict__) - 1 + len(self._extra)
    '''repr'''
    def __repr__(self):
        core = {k: v for k, v in self.__dict__.items() if k != '_extra'}
        merged = {**core, **self._extra}
        return f"DatasetConfig({merged})"
    '''copy'''
    def copy(self):
        return copy.deepcopy(self)