# Tutorials


## Learn about Config

We incorporate modular design into our config system, which is convenient to conduct various experiments. 

#### Config File Structure

Now, there are 2 basic component types under "configs/\_base\_", *i.e.*, datasets and dataloaders, which are responsible for loading various datasets with different runtime settings (*e.g.*, batch size, image size, data augmentation, to name a few).
For example, to train FCN segmentor on Pascal VOC dataset (assuming that we hope the total batch size is 16 and the training image size is 512x512), you can import the corresponding pre-defined config like this,
```python
from .._base_ import DATASET_CFG_VOCAUG_512x512, DATALOADER_CFG_BS16
```
Then, modify "SEGMENTOR_CFG" in the corresponding method config file (*e.g.*, "fcn_resnet50os16_voc.py") as follows,
```python
# modify dataset config
SEGMENTOR_CFG['dataset'] = DATASET_CFG_VOCAUG_512x512.copy()
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = DATALOADER_CFG_BS16.copy()
```
From this, you are not required to define the same dataloader and dataset configs over and over again if you just want to follow the conventional training and testing settings of various datasets to train your segmentors.

Next, we talk about config files in specific method directories (*e.g.*, "fcn" directory), there is also one and only one "base_cfg.py" used to define some necessary configs for these methods.
Like loading configs from datasets and dataloaders in "configs/\_base\_", you can also import the "base_cfg.py" and simply modify some key values in "SEGMENTOR_CFG" to customize and train the corresponding segmentor.
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
You can refer to "ssseg/configs" for more examples about creating a valid config file.