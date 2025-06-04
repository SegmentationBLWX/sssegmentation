'''initialize'''
import os
import importlib
from .default_segmentor import SegmentorConfig


'''register segmentors'''
REGISTERED_SEGMENTOR_CONFIGS = {}
for fname in os.listdir(os.path.dirname(__file__)):
    if fname.endswith(".py") and fname != "__init__.py" and fname != "default_segmentor.py":
        module_name = fname[:-3]
        module = importlib.import_module(f".{module_name}", package=__name__)
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, SegmentorConfig):
                REGISTERED_SEGMENTOR_CONFIGS[attr_name] = attr