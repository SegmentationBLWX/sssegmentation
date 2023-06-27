# Tutorials


## Learn about Config

We incorporate modular design into our config system, which is convenient to conduct various experiments. 

#### Config File Structure

Now, there are 2 basic component types under "configs/_base_", *i.e.*, datasets and dataloaders, which are responsible for loading various datasets with different settings.
For example, to train FCN segmentor on Pascal VOC dataset, you can import them like this,
```python
from .._base_ import DATASET_CFG_VOCAUG_512x512, DATALOADER_CFG_BS16
```
Then, modify "SEGMENTOR_CFG" in the corresponding config file as follows,
```python
# modify dataset config
SEGMENTOR_CFG['dataset'] = DATASET_CFG_VOCAUG_512x512.copy()
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = DATALOADER_CFG_BS16.copy()
```

Then, speaking of specific methods, there is also one and only one "base_cfg.py" used to define some necessary information for these methods.
Like loading configs from datasets and dataloaders in "configs/_base_", you can also import the "base_cfg.py" and modify some keys in "SEGMENTOR_CFG" to customize your segmentors.
For instance, to customize FCN with ResNet-50-D16 backbone and train it on Pascal VOC dataset, you can create a config file in "fcn" directory, named "fcn_resnet50os16_voc.py" and write in some contents like this,
```python
'''fcn_resnet50os16_voc'''
import copy
from .base_cfg import SEGMENTOR_CFG
from .._base_ import DATASET_CFG_VOCAUG_512x512, DATALOADER_CFG_BS16


# deepcopy
SEGMENTOR_CFG = copy.deepcopy(SEGMENTOR_CFG)
# modify dataset config
SEGMENTOR_CFG['dataset'] = DATASET_CFG_VOCAUG_512x512.copy()
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = DATALOADER_CFG_BS16.copy()
# modify scheduler config
SEGMENTOR_CFG['scheduler']['max_epochs'] = 60
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 21
SEGMENTOR_CFG['backbone'] = {
    'type': 'ResNet', 'depth': 50, 'structure_type': 'resnet50conv3x3stem',
    'pretrained': True, 'outstride': 16, 'use_conv3x3_stem': True, 'selected_indices': (2, 3),
}
SEGMENTOR_CFG['work_dir'] = 'fcn_resnet50os16_voc'
SEGMENTOR_CFG['logfilepath'] = 'fcn_resnet50os16_voc/fcn_resnet50os16_voc.log'
SEGMENTOR_CFG['resultsavepath'] = 'fcn_resnet50os16_voc/fcn_resnet50os16_voc_results.pkl'
```