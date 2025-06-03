'''default_dataloader'''
from typing import Any, Dict
from collections.abc import MutableMapping
from dataclasses import dataclass, field, asdict


'''DataloaderConfig'''
@dataclass
class DataloaderConfig(MutableMapping):
    expected_total_train_bs_for_assert: int
    auto_adapt_to_expected_train_bs: bool
    train: Dict[str, Any] = field(default_factory=dict)
    test: Dict[str, Any] = field(default_factory=dict)
    '''getitem'''
    def __getitem__(self, key):
        return asdict(self)[key]
    '''setitem'''
    def __setitem__(self, key, value):
        setattr(self, key, value)
    '''delitem'''
    def __delitem__(self, key):
        delattr(self, key)
    '''iter'''
    def __iter__(self):
        return iter(asdict(self))
    '''len'''
    def __len__(self):
        return len(asdict(self))
    '''repr'''
    def __repr__(self):
        return f"DataloaderConfig({asdict(self)})"