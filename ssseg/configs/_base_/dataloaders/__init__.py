'''initialize'''
import os
import importlib
from .default_dataloader import DataloaderConfig


'''auto import all'''
__all__ = []
for fname in os.listdir(os.path.dirname(__file__)):
    if fname.endswith(".py") and fname != "__init__.py":
        module_name = fname[:-3]
        module = importlib.import_module(f".{module_name}", package=__name__)
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, DataloaderConfig):
                globals()[attr_name] = attr
                __all__.append(attr_name)