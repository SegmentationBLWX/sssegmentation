# Tutorials

In this chapter, we will provide detailed tutorials to help the users learn how to use SSSegmentation.


## Learn about Config

We incorporate modular design into our config system, which is convenient to conduct various experiments. 

#### Config File Structure

Now, there are 2 basic component types under `ssseg/configs/_base_`, *i.e.*, `datasets` and `dataloaders`, which are responsible for defining configs of various datasets with different runtime settings (*e.g.*, batch size, image size, data augmentation, to name a few).

For example, to train FCN segmentor on Pascal VOC dataset (assuming that we hope the total batch size is 16 and the training image size is 512x512), you can import the corresponding pre-defined config like this,

```python
from .._base_ import DATASET_CFG_VOCAUG_512x512, DATALOADER_CFG_BS16
```

Then, modify `SEGMENTOR_CFG` in the corresponding method config file (*e.g.*, `ssseg/configs/fcn/fcn_resnet50os16_voc.py`) as follows,

```python
# modify dataset config
SEGMENTOR_CFG['dataset'] = DATASET_CFG_VOCAUG_512x512.copy()
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = DATALOADER_CFG_BS16.copy()
```

From this, you are not required to define the same dataloader and dataset configs over and over again if you just want to follow the conventional training and testing settings of various datasets to train your segmentors.

Next, we talk about config files in specific segmentation algorithm directories (*e.g.*, `ssseg/configs/fcn` directory). 
There is also one and only one `base_cfg.py` used to define some necessary configs for the corresponding algorithm (*e.g.*, `ssseg/configs/fcn/base_cfg.py` is used to define the basic configs for FCN segmentation algorithm).
Like loading configs from `ssseg/configs/_base_`, you can also import the `base_cfg.py` and simply modify some key values in `SEGMENTOR_CFG` to customize and train the corresponding segmentor.

For instance, to customize FCN with ResNet-50-D16 backbone and train it on Pascal VOC dataset, you can create a config file in `ssseg/configs/fcn` directory, named `fcn_resnet50os16_voc.py` and write in some contents like this,

```python
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

After that, you can train this segmentor with the following command,

```sh
bash scripts/dist_train.sh 4 ssseg/configs/fcn/fcn_resnet50os16_voc.py
```

How relaxing and enjoyable! Maybe, you can read more config examples in `ssseg/configs` to help you learn about how to create a valid config file in SSSegmentation.

#### An Example of PSPNet

To help the users have a basic idea of a complete config and the modules in SSSegmentation, we make brief comments on the config of PSPNet using ResNet-101-D8 as the following,

```python
import os

# dataset configs
DATASET_CFG_ADE20k_512x512 = {
    'type': 'ADE20kDataset', # the dataset type used to instance the specific dataset class in builder.py defined in "ssseg/modules/datasets/builder.py"
    'rootdir': os.path.join(os.getcwd(), 'ADE20k'), # the corresponding dataset path you download
    'train': {
        'set': 'train', # train the models with the train set defined in the dataset
        'data_pipelines': [
            ('Resize', {'output_size': (2048, 512), 'keep_ratio': True, 'scale_range': (0.5, 2.0)}),
            ('RandomCrop', {'crop_size': (512, 512), 'one_category_max_ratio': 0.75}),
            ('RandomFlip', {'flip_prob': 0.5}),
            ('PhotoMetricDistortion', {}),
            ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
            ('ToTensor', {}),
            ('Padding', {'output_size': (512, 512), 'data_type': 'tensor'}),
        ], # define some pre-processing operations for the loaded images and segmentation targets during training, you can refer to "ssseg/modules/datasets/pipelines" for more details
    },
    'test': {
        'set': 'val', # test the models with the validation set defined in the dataset
        'data_pipelines': [
            ('Resize', {'output_size': (2048, 512), 'keep_ratio': True, 'scale_range': None}),
            ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
            ('ToTensor', {}),
        ], # define some pre-processing operations for the loaded images and segmentation targets during testing, you can refer to "ssseg/modules/datasets/pipelines" for more details
    }
}
DATALOADER_CFG_BS16 = {
    'expected_total_train_bs_for_assert': 16, # it is defined for asserting whether the users adopt the correct batch size for training the models
	'auto_adapt_to_expected_train_bs': True, # if set Ture, the "expected_total_train_bs_for_assert" will be used to determine the value of "batch_size_per_gpu" rather than using the specified value in the config
    'train': {
        'batch_size_per_gpu': 2, # number of images in each gpu during training
        'num_workers_per_gpu': 2, # number of workers for dataloader in each gpu during training
        'shuffle': True, # whether to shuffle the image order during training
        'pin_memory': True, # whether to achieve higher bandwidth between the host and the device using pinned memory during training
        'drop_last': True, # whether to drop out the last images which cannot form a batch size of "expected_total_train_bs_for_assert" during training
    },
    'test': {
        'batch_size_per_gpu': 1, # number of images in each gpu during testing
        'num_workers_per_gpu': 2, # number of workers for dataloader in each gpu during testing
        'shuffle': False, # whether to shuffle the image order during testing
        'pin_memory': True, # whether to achieve higher bandwidth between the host and the device using pinned memory during testing
        'drop_last': False, # whether to drop out the last images which cannot form a batch size of "expected_total_train_bs_for_assert" during testing
    }
}
SEGMENTOR_CFG = {
    'type': 'PSPNet', # the segmentor type defined in "ssseg/modules/models/segmentors/builder.py"
    'num_classes': -1, # number of classes in the dataset
    'benchmark': True, # set True for speeding up training
    'align_corners': False, # align_corners in torch.nn.functional.interpolate
    'backend': 'nccl', # backend for DDP training and testing
    'work_dir': 'ckpts', # directory used to save checkpoints and training and testing logs
    'logfilepath': '', # file path to record the training and testing logs
    'log_interval_iterations': 50, # print training log after "log_interval_iterations" iterations
    'eval_interval_epochs': 10, # evaluate models after "eval_interval_epochs" epochs
    'save_interval_epochs': 1, # save the checkpoints of models after "save_interval_epochs" epochs
    'resultsavepath': '', # path used to save the testing results used in "ssseg/test.py"
    'norm_cfg': {'type': 'SyncBatchNorm'}, # config for normalization layer in the segmentor
    'act_cfg': {'type': 'ReLU', 'inplace': True}, # config for activation layer in the segmentor
    'backbone': {
        'type': 'ResNet', 'depth': 101, 'structure_type': 'resnet101conv3x3stem',
        'pretrained': True, 'outstride': 8, 'use_conv3x3_stem': True, 'selected_indices': (2, 3),
    }, # define backbones, refer to "ssseg/modules/models/backbones/builder.py" for more details
    'head': {
        'in_channels': 2048, 'feats_channels': 512, 'pool_scales': [1, 2, 3, 6], 'dropout': 0.1,
    }, # define decoder heads, refer to "ssseg/modules/models/segmentors" for more details
    'auxiliary': {
        'in_channels': 1024, 'out_channels': 512, 'dropout': 0.1,
    }, # define auxiliary decoder, refer to "ssseg/modules/models/segmentors/base/base.py" for more details
    'losses': {
        'loss_aux': {'CrossEntropyLoss': {'scale_factor': 0.4, 'ignore_index': 255, 'reduction': 'mean'}},
        'loss_cls': {'CrossEntropyLoss': {'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'}},
    }, # define objective functions, refer to "ssseg/modules/models/losses/builder.py" for more details
    'inference': {
        'mode': 'whole',
        'opts': {}, 
        'tricks': {
            'multiscale': [1], 'flip': False, 'use_probs_before_resize': False,
        }
    }, # define the inference config, refer to "ssseg/test.py" for more details
    'scheduler': {
        'type': 'PolyScheduler', 'max_epochs': 0, 'power': 0.9,
        'optimizer': {
            'type': 'SGD', 'lr': 0.01, 'momentum': 0.9, 'weight_decay': 5e-4, 'params_rules': {},
        }
    }, # define the scheduler configs, refer to "ssseg/modules/models/schedulers/builder.py" for more details
    'dataset': DATASET_CFG_ADE20k_512x512, # define the dataset configs
    'dataloader': DATALOADER_CFG_BS16, # define the dataloader configs
}
```

In the following sections, we will give more explanations and examples about each module specified in the config above.


## Customize Datasets

Dataset classes in SSSegmentation have two functions: (1) load data information after data preparation and (2) construct `sample_meta` for the subsequent segmentor training and testing.

The type of `sample_meta` is `dict` which includes several keys:

- `image`: The loaded image data, 
- `seg_target`: The loaded ground truth segmentation mask data,
- `width` and `height`: The original size of the image (*i.e.*, the image size before pre-processing by the data transforms),
- `id`: The image id of the loaded image data.

Thanks to the modular design in SSSegmentation, we can simply modify the configs in `SEGMENTOR_CFG['dataset']` to train one segmentor on various datasets.

#### Dataset Config Structure

An example of dataset config is as follows,

```python
import os

DATASET_CFG_ADE20k_512x512 = {
    'type': 'ADE20kDataset',
    'rootdir': os.path.join(os.getcwd(), 'ADE20k'),
    'train': {
        'set': 'train',
        'data_pipelines': [
            ('Resize', {'output_size': (2048, 512), 'keep_ratio': True, 'scale_range': (0.5, 2.0)}),
            ('RandomCrop', {'crop_size': (512, 512), 'one_category_max_ratio': 0.75}),
            ('RandomFlip', {'flip_prob': 0.5}),
            ('PhotoMetricDistortion', {}),
            ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
            ('ToTensor', {}),
            ('Padding', {'output_size': (512, 512), 'data_type': 'tensor'}),
        ],
    },
    'test': {
        'set': 'val',
        'data_pipelines': [
            ('Resize', {'output_size': (2048, 512), 'keep_ratio': True, 'scale_range': None}),
            ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
            ('ToTensor', {}),
        ],
    }
}
```

where `type` denotes the dataset you want to train on. Now, SSSegmentation supports the following dataset types,

```python
REGISTERED_MODULES = {
    'BaseDataset': BaseDataset, 'VOCDataset': VOCDataset, 'PascalContext59Dataset': PascalContext59Dataset, 'PascalContextDataset': PascalContextDataset,
    'COCODataset': COCODataset, 'COCOStuff10kDataset': COCOStuff10kDataset, 'COCOStuffDataset': COCOStuffDataset, 'CIHPDataset': CIHPDataset,
    'LIPDataset': LIPDataset, 'ATRDataset': ATRDataset, 'MHPv1Dataset': MHPv1Dataset, 'MHPv2Dataset': MHPv2Dataset, 'SuperviselyDataset': SuperviselyDataset,
    'HRFDataset': HRFDataset, 'ChaseDB1Dataset': ChaseDB1Dataset, 'STAREDataset': STAREDataset, 'DRIVEDataset': DRIVEDataset, 'SBUShadowDataset': SBUShadowDataset,
    'VSPWDataset': VSPWDataset, 'ADE20kDataset': ADE20kDataset, 'DarkZurichDataset': DarkZurichDataset, 'NighttimeDrivingDataset': NighttimeDrivingDataset,
    'CityScapesDataset': CityScapesDataset,
}
```

The keyword `rootdir` is used to specify the data set path. It is recommended to symlink the dataset root to `$SSSEGMENTATION/` or directly run `bash scripts/prepare_datasets.sh $DATASETNAME` in `$SSSEGMENTATION/` so that you don't need to modify the default `rootdir` for each dataset.

The keyword `train` and `test` are used to specify the configs for model training and testing. 
And the value type of `SEGMENTOR_CFG['dataset']['train']` and `SEGMENTOR_CFG['dataset']['test']` is `dict`.
Specifically, `set` means a certain division of the data set, usually including train set (set as `train`), validation set (set as `val`) and test set (set as `test`). 
`data_pipelines` is used to define data transforms to pre-process `sample_meta` before feeding into the models, more details please refer to [Customize Data Pipelines](https://sssegmentation.readthedocs.io/en/latest/Tutorials.html#customize-data-pipelines).

The other arguments supported in `SEGMENTOR_CFG['dataset']` is listed as follows,

- `repeat_times`: The default value is 1, if increase it in the config, the appeared times of one image in an epoch will increase accordingly,
- `evalmode`: `local` or `server`, `server` denotes `seg_target` will be set as `None`.

If the users want to learn more about this part, it is recommended that you could jump to the [`ssseg/modules/datasets` directory](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/datasets) in SSSegmentation to read the source codes of dataset classes.

#### Customize Data Pipelines

Constructing data pipelines is used to preprocess the input data (*e.g.*, images and segmentation masks) for the following training and testing of the segmentors.

Specifically, it is defined at,

- `SEGMENTOR_CFG['dataset']['train']['data_pipelines']`: The constructed data pipelines for training,
- `SEGMENTOR_CFG['dataset']['test']['data_pipelines']`: The constructed data pipelines for testing.

The value of the `data_pipelines` should be a `list` like following,

```python
SEGMENTOR_CFG['dataset']['train']['data_pipelines'] = [
    ('Resize', {'output_size': (2048, 512), 'keep_ratio': True, 'scale_range': (0.5, 2.0)}),
    ('RandomCrop', {'crop_size': (512, 512), 'one_category_max_ratio': 0.75}),
    ('RandomFlip', {'flip_prob': 0.5}),
    ('PhotoMetricDistortion', {}),
    ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
    ('ToTensor', {}),
    ('Padding', {'output_size': (512, 512), 'data_type': 'tensor'}),
]
```

And each item in the list should be a `tuple` or `dict`. For example, it could be,

```
# tuple
('Resize', {'output_size': (2048, 512), 'keep_ratio': True, 'scale_range': (0.5, 2.0)})
# dict
{'type': 'Resize', 'output_size': (2048, 512), 'keep_ratio': True, 'scale_range': (0.5, 2.0)}
```

where `Resize` means a data transform method defined in [`ssseg/modules/datasets/pipelines/transforms.py`](https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/datasets/pipelines/transforms.py) and other values denote for the arguments for instancing the corresponding data transform method.

Here is a list of supported data transform methods,

```python
REGISTERED_MODULES = {
    'Resize': Resize, 'RandomCrop': RandomCrop, 'RandomFlip': RandomFlip, 'RandomRotation': RandomRotation, 'EdgeExtractor': EdgeExtractor,
    'PhotoMetricDistortion': PhotoMetricDistortion, 'Padding': Padding, 'ToTensor': ToTensor, 'ResizeShortestEdge': ResizeShortestEdge,
    'Normalize': Normalize, 'RandomChoiceResize': RandomChoiceResize, 'Rerange': Rerange, 'CLAHE': CLAHE, 'RandomCutOut': RandomCutOut, 
    'AlbumentationsWrapper': AlbumentationsWrapper, 'RGB2Gray': RGB2Gray, 'AdjustGamma': AdjustGamma,
}
```

To learn the functions of each data transform method, please check the source codes in [`ssseg/modules/datasets/pipelines/transforms.py`](https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/datasets/pipelines/transforms.py) by yourselves.

It is worth mentioning that SSSegmentation provides `AlbumentationsWrapper` to make the users leverage the data augmentation algorithms implemented in [albumentations](https://albumentations.ai/docs/).
Here is an example of calling `AlbumentationsWrapper`,

```python
SEGMENTOR_CFG['dataset']['train']['data_pipelines'] = [
    ('AlbumentationsWrapper', {'albu_cfg': {'type': 'RandomCrop', 'width': 256, 'height': 256}}),
    ('AlbumentationsWrapper', {'albu_cfg': {'type': 'HorizontalFlip', 'p': 0.5}}),
    ('AlbumentationsWrapper', {'albu_cfg': {'type': 'RandomBrightnessContrast', 'p': 0.2}}),
]
```

Finally, if you want to define the data transform method by yourselves during developing, you can first write the transform method like following,

```
class RGB2Gray(object):
    def __init__(self, out_channels=None, weights=(0.299, 0.587, 0.114)):
        # assert
        assert isinstance(weights, collections.abc.Sequence)
        assert out_channels is None or out_channels > 0
        for item in weights: assert isinstance(item, (float, int))
        # set attributes
        self.weights = weights
        self.out_channels = out_channels
    '''call'''
    def __call__(self, sample_meta):
        sample_meta = self.rgb2gray('image', sample_meta, self.weights, self.out_channels)
        return sample_meta
    '''rgb2gray'''
    @staticmethod
    def rgb2gray(key, sample_meta, weights, out_channels):
        if key not in sample_meta: return sample_meta
        # assert
        assert len(sample_meta[key].shape) == 3
        assert sample_meta[key].shape[2] == len(weights)
        # apply
        weights = np.array(weights).reshape((1, 1, -1))
        sample_meta[key] = (sample_meta[key] * weights).sum(2, keepdims=True)
        if out_channels is None:
            sample_meta[key] = sample_meta[key].repeat(weights.shape[2], axis=2)
        else:
            sample_meta[key] = sample_meta[key].repeat(out_channels, axis=2)
        # return
        return sample_meta
```

Then, import `DataTransformBuilder` and register this method,

```python
from ssseg.modules import DataTransformBuilder

data_transformer_builder = DataTransformBuilder()
data_transformer_builder.register('RGB2Gray', RGB2Gray)
```

From this, you can call `data_transformer_builder.build` to build your own defined transform algorithms as well as the original supported data transform methods.

#### Add New Custom Dataset

SSSegmentation provides `BaseDataset` class to help the users quickly add a new custom dataset. 

Specifically, you can directly inherit this class to define your own dataset class. Here is an example code to add `SuperviselyDataset` dataset in SSSegmentation [`ssseg/modules/datasets` directory](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/datasets) by using `BaseDataset`,

```
import os
import pandas as pd
from .base import BaseDataset

'''SuperviselyDataset'''
class SuperviselyDataset(BaseDataset):
    num_classes = 2
    classnames = ['__background__', 'person']
    palette = [(0, 0, 0), (255, 0, 0)]
    clsid2label = {255: 1}
    assert num_classes == len(classnames) and num_classes == len(palette)
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(SuperviselyDataset, self).__init__(mode=mode, logger_handle=logger_handle, dataset_cfg=dataset_cfg)
        # obtain the dirs
        rootdir = dataset_cfg['rootdir']
        self.image_dir = os.path.join(rootdir, 'Images', dataset_cfg['set'])
        self.ann_dir = os.path.join(rootdir, 'Anno-Person', dataset_cfg['set'])
        # obatin imageids
        df = pd.read_csv(os.path.join(rootdir, dataset_cfg['set']+'.txt'), names=['imageids'])
        self.imageids = df['imageids'].values
        self.imageids = [str(_id) for _id in self.imageids]
```

In `__init__`, you are required to set some attributes including:

- `image_dir`: Image file directory,
- `ann_dir`: Annotation file directory (*i.e.*, the directory used to save ground truth segmentation masks),
- `image_ext`: Image file extension, default value is `.jpg`,
- `ann_ext`: Annotation file extension, default value is `.png`,
- `imageids`: The image file names.

Besides, `clsid2label` could be set for transferring the label ids in ground truth segmentation mask to continuous training ids automatically during training. 
For example, `clsid2label = {10: 1}` means if the class id in ground truth segmentation mask is 10, they will be set as class id 1 before feeding into the models.
Other attributes `classnames` is used to set the class names in the dataset, `palette` is used to set the colors for each class during [Inference with Segmentors Integrated in SSSegmentation](https://sssegmentation.readthedocs.io/en/latest/QuickRun.html#inference-with-segmentors-integrated-in-sssegmentation) and `num_classes` represents the number of the classes existed in the dataset.

After that, you should add this custom dataset class in [`ssseg/modules/datasets/builder.py`](https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/datasets/builder.py) if you want to use it by simply modifying `SEGMENTOR_CFG['dataset']`.
Of course, you can also register this custom dataset by the following codes,

```python
from ssseg.modules import DatasetBuilder

dataset_builder = DatasetBuilder()
dataset_builder.register('SuperviselyDataset', SuperviselyDataset)
```

From this, you can also call `dataset_builder.build` to build your own defined dataset class as well as the original supported data classes.

Finally, the users could jump to the [`ssseg/modules/datasets` directory](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/datasets) in SSSegmentation to read more source codes of the supported dataset classes and thus better learn how to customize the dataset classes in SSSegmentation.


## Customize Backbones

Backbone is the image encoder that transforms an image to feature maps, such as a ResNet-50 without the last fully connected layer.

#### Backbone Config Structure

An example of backbone config is as follows,

```python
SEGMENTOR_CFG['backbone'] = {
    'type': 'ResNet', 'depth': 101, 'structure_type': 'resnet101conv3x3stem',
    'pretrained': True, 'outstride': 16, 'use_conv3x3_stem': True, 'selected_indices': (2, 3),
}
```

where `type` denotes the backbone network you want to employ. Now, SSSegmentation supports the following backbone types,

```python
REGISTERED_MODULES = {
    'UNet': UNet, 'BEiT': BEiT, 'CGNet': CGNet, 'HRNet': HRNet, 'MobileViT': MobileViT, 'MobileViTV2': MobileViTV2,
    'ERFNet': ERFNet, 'ResNet': ResNet, 'ResNeSt': ResNeSt, 'PCPVT': PCPVT, 'MobileSAMTinyViT': MobileSAMTinyViT, 
    'SVT': SVT, 'FastSCNN': FastSCNN, 'ConvNeXt': ConvNeXt, 'BiSeNetV1': BiSeNetV1, 'MAE': MAE, 'SAMViT': SAMViT,
    'BiSeNetV2': BiSeNetV2, 'SwinTransformer': SwinTransformer, 'VisionTransformer': VisionTransformer,
    'MixVisionTransformer': MixVisionTransformer, 'TIMMBackbone': TIMMBackbone, 'ConvNeXtV2': ConvNeXtV2,
    'MobileNetV2': MobileNetV2, 'MobileNetV3': MobileNetV3, 
}
```

The other arguments in `SEGMENTOR_CFG['backbone']` are set for instancing the corresponding backbone network. 

Here we also list some common arguments and their explanation,

- `structure_type`: The structure type of the specified backbone network, *e.g.*, `resnet101conv3x3stem` means ResNet-101 using three 3x3 convolutions as the stem layer, it is useful if you want to load the pretrained backbone weights automatically,
- `pretrained`: Whether to load the pretrained backbone weights,
- `pretrained_model_path`: If you set `pretrained_model_path` as `None` and `pretrained` as `True`, SSSegmentation will load the pretrained backbone weights automatically, otherwise, load the pretrained backbone weights from the path specified by `pretrained_model_path`,
- `out_indices`: Generally, a backbone network can be divided into several stages, `out_indices` is used to specify whether return the feature maps outputted by the corresponding backbone stage,
- `norm_cfg`: The config of normalization layer, it should be a `dict`, refer to [customize-normalizations](https://sssegmentation.readthedocs.io/en/latest/Tutorials.html#customize-normalizations) more details,
- `act_cfg`: The config of activation layer, it should be a `dict`, refer to [customize-activations](https://sssegmentation.readthedocs.io/en/latest/Tutorials.html#customize-activations) more details.

To learn more about how to set the specific arguments for each backbone type, you can jump to [`ssseg/modules/models/backbones` directory](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/backbones) to check the source codes of each backbone network.

#### Add New Custom Backbone

If the users want to add a new custom backbone, you should first create a new file in [`ssseg/modules/models/backbones` directory](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/backbones), *e.g.*, [`ssseg/modules/models/backbones/mobilenet.py`](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/backbones/mobilenet.py).

Then, you can define the backbone module in this file by yourselves, *e.g.*,

```python
import torch.nn as nn

'''MobileNet'''
class MobileNet(nn.Module):
    def __init__(self, arg1, arg2):
        pass
    def forward(self, x):
        pass
```

After that, you should add this custom backbone class in [`ssseg/modules/models/backbones/builder.py`](https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/backbones/builder.py) if you want to use it by simply modifying `SEGMENTOR_CFG['backbone']`.
Of course, you can also register this custom backbone by the following codes,

```python
from ssseg.modules import BackboneBuilder

backbone_builder = BackboneBuilder()
backbone_builder.register('MobileNet', MobileNet)
```

From this, you can also call `backbone_builder.build` to build your own defined backbone class as well as the original supported backbone classes.

Finally, the users could jump to the [`ssseg/modules/models/backbones` directory](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/backbones) in SSSegmentation to read more source codes of the supported backbone classes and thus better learn how to customize the backbone classes in SSSegmentation.


## Customize Losses

Loss is utilized to define the objective functions for the segmentation framework, such as the [Cross Entropy Loss](https://en.wikipedia.org/wiki/Cross-entropy).

#### Loss Config Structure

An example of loss config is as follows,

```python
SEGMENTOR_CFG['losses'] = {
    'loss_aux': {'CrossEntropyLoss': {'scale_factor': 0.4, 'ignore_index': 255, 'reduction': 'mean'}},
    'loss_cls': {'CrossEntropyLoss': {'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'}},
}
```

It is a `dict` including several keys like `loss_aux` and `loss_cls`, which is designed for distinguishing the loss config utilized in head or auxiliary head.
The value corresponding to each key is also a `dict` and in this `dict`, `key` denotes the objective function type you want to adopt and the corresponding value denotes the arguments for instancing this objective function.

In the example of loss config above, `SEGMENTOR_CFG['losses']['loss_cls']` is used to build losses in head and `SEGMENTOR_CFG['losses']['loss_aux']` is used to build losses in auxiliary head.
`CrossEntropyLoss` represents the objective function type you want to adopt and `{'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'}` contains the arguments for instancing the `CrossEntropyLoss`.

Now, SSSegmentation supports the following loss types,

```python
REGISTERED_MODULES = {
    'L1Loss': L1Loss, 'DiceLoss': DiceLoss, 'KLDivLoss': KLDivLoss, 'LovaszLoss': LovaszLoss,
    'CrossEntropyLoss': CrossEntropyLoss, 'SigmoidFocalLoss': SigmoidFocalLoss,
    'CosineSimilarityLoss': CosineSimilarityLoss, 'BinaryCrossEntropyLoss': BinaryCrossEntropyLoss,
}
```

Here we also list some common arguments and their explanation,

- `scale_factor`: The return loss value is equal to the original loss times `scale_factor`,
- `ignore_index`: Specifies a target value that is ignored and does not contribute to the input gradient,
- `lowest_loss_value`: If `lowest_loss_value` is set, the return loss value is equal to `min(lowest_loss_value, scale_factor * original loss)`, this argument is designed according to the paper [Do We Need Zero Training Loss After Achieving Zero Training Error? - ICML 2020](https://arxiv.org/pdf/2002.08709.pdf).

To learn more about how to set the specific arguments for each loss function, you can jump to [`ssseg/modules/models/losses` directory](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/losses) to check the source codes of each loss function.

#### Add New Custom Loss

If the users want to add a new custom loss, you should first create a new file in [`ssseg/modules/models/losses` directory](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/losses), *e.g.*, [`ssseg/modules/models/losses/klloss.py`](https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/losses/klloss.py).

Then, you can define the loss function in this file by yourselves, *e.g.*,

```python
import torch.nn as nn

'''KLDivLoss'''
class KLDivLoss(nn.Module):
    def __init__(self, arg1, arg2):
        pass
    def forward(self, prediction, target):
        pass
```

After that, you should add this custom loss class in [`ssseg/modules/models/losses/builder.py`](https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/losses/builder.py) if you want to use it by simply modifying `SEGMENTOR_CFG['losses']`.
Of course, you can also register this custom loss function by the following codes,

```python
from ssseg.modules import LossBuilder

loss_builder = LossBuilder()
loss_builder.register('KLDivLoss', KLDivLoss)
```

From this, you can also call `loss_builder.build` to build your own defined loss class as well as the original supported loss classes.

Finally, the users could jump to the [`ssseg/modules/models/losses` directory](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/losses) in SSSegmentation to read more source codes of the supported loss classes and thus better learn how to customize the loss classes in SSSegmentation.


## Customize Schedulers

Scheduler provides several methods to adjust the learning rate based on the number of epochs or iterations.

#### Scheduler Config Structure

An example of scheduler config is as follows,

```python
SEGMENTOR_CFG['scheduler'] = {
    'type': 'PolyScheduler', 'max_epochs': 0, 'power': 0.9,
    'optimizer': {
        'type': 'SGD', 'lr': 0.01, 'momentum': 0.9, 'weight_decay': 5e-4, 'params_rules': {},
    }
}
```

where `type` denotes the scheduler you want to utilize during training. Now, SSSegmentation supports the following scheduler types,

```python
REGISTERED_MODULES = {
    'PolyScheduler': PolyScheduler,
}
```

The other arguments in `SEGMENTOR_CFG['scheduler']` are set for instancing the corresponding scheduler, where `SEGMENTOR_CFG['scheduler']['optimizer']` is the optimizer config used to build a optimizer for model training.
The detailed instruction about building optimizer in SSSegmentation please refer to [`Customize Optimizers`](https://sssegmentation.readthedocs.io/en/latest/Tutorials.html#customize-optimizers).

To learn more about how to set the specific arguments for each scheduler, you can jump to [`ssseg/modules/models/schedulers` directory](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/schedulers) to check the source codes of each scheduler class.

#### Customize Optimizers

Optimizer is used to define the process of adjusting model parameters to reduce model error in each training step, such as the [Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent).

Specifically, an example of optimizer config used to construct an optimizer for model training could be,

```python
SEGMENTOR_CFG['scheduler']['optimizer'] = {
    'type': 'SGD', 'lr': 0.01, 'momentum': 0.9, 'weight_decay': 5e-4, 'params_rules': {},
}
```

where `type` denotes the optimizer you want to utilize during training. Here is a list of supported optimizer types,

```python
REGISTERED_MODULES = {
    'SGD': optim.SGD, 'Adam': optim.Adam, 'AdamW': optim.AdamW, 'Adadelta': optim.Adadelta,
}
```

The other arguments in `SEGMENTOR_CFG['scheduler']['optimizer']` are set for instancing the corresponding optimizer.

Among these arguments, `params_rules` could be set for implementing some training tricks. 
For example, if you want to train the backbone network and the decoder layer with different learning rates, you can set `params_rules` as following,

```python
SEGMENTOR_CFG['scheduler']['optimizer']['params_rules'] = {
    'backbone_net': dict(lr_multiplier=0.1, wd_multiplier=1.0),
}
```

where `lr_multiplier = 0.1` means the learning rate of backbone network is one-tenth of the decoder layer.

And if you want to set the weight decay of some layers in the segmentor as zeros, you can set `params_rules` as following,

```python
SEGMENTOR_CFG['scheduler']['optimizer']['params_rules'] = {
    'absolute_pos_embed': dict(wd_multiplier=0.),
    'relative_position_bias_table': dict(wd_multiplier=0.),
    'norm': dict(wd_multiplier=0.),
}
```

You can refer to the [`ssseg/configs` directory](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/configs) for more examples about implementing some training tricks by setting `params_rules`.

Finally, if you want to customize the optimizer by yourselves during developing, you should first create a new file in [`ssseg/modules/models/optimizers` directory](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/optimizers), *e.g.*, `ssseg/modules/models/optimizers/sgd.py`.

Then, you can define the optimization algorithm in this file by yourselves, *e.g.*,

```python
class SGD():
    def __init__(self, arg1, arg2):
        pass
```

After that, you should add this custom optimization algorithm in [`ssseg/modules/models/optimizers/builder.py`](https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/optimizers/builder.py) if you want to use it by simply modifying `SEGMENTOR_CFG['scheduler']['optimizer']`.
Of course, you can also register this custom optimization algorithm by the following codes,

```python
from ssseg.modules import OptimizerBuilder

optimizer_builder = NormalizationBuilder()
optimizer_builder.register('SGD', SGD)
```

From this, you can also call `optimizer_builder.build` to build your own defined optimization algorithms as well as the original supported optimization algorithms.

#### Add New Custom Scheduler

SSSegmentation provides `BaseScheduler` class to help the users quickly add a new custom scheduler.

Specifically, if the users want to add a new custom scheduler, you should first create a new file in [`ssseg/modules/models/schedulers` directory](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/schedulers), *e.g.*, [`ssseg/modules/models/schedulers/polyscheduler.py`](https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/schedulers/polyscheduler.py).

Then, you can define the scheduler in this file by inheriting `BaseScheduler`, *e.g.*,

```python
from .basescheduler import BaseScheduler

'''PolyScheduler'''
class PolyScheduler(BaseScheduler):
    def __init__(self, arg1, arg2):
        pass
    def updatelr(self):
        pass
```

where `updatelr` function is used to define the learning rate adjust strategy.

After that, you should add this custom scheduler class in [`ssseg/modules/models/schedulers/builder.py`](https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/schedulers/builder.py) if you want to use it by simply modifying `SEGMENTOR_CFG['scheduler']`.
Of course, you can also register this custom scheduler by the following codes,

```python
from ssseg.modules import SchedulerBuilder

scheduler_builder = SchedulerBuilder()
scheduler_builder.register('PolyScheduler', PolyScheduler)
```

From this, you can also call `scheduler_builder.build` to build your own defined schedulers as well as the original supported schedulers.

Finally, the users could jump to the [`ssseg/modules/models/schedulers` directory](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/schedulers) in SSSegmentation to read more source codes of the supported schedulers and thus better learn how to customize the schedulers in SSSegmentation.


## Customize Segmentors

Segmentor will process the feature maps outputted by the backbone network, and transforms the feature maps to a predicted segmentation mask using a decoder head (*e.g.*, [Deeplabv3](https://arxiv.org/pdf/1706.05587.pdf) and [IDRNet](https://arxiv.org/pdf/2310.10755.pdf)).

Here is an example of head config,

```python
SEGMENTOR_CFG['head'] = {
    'in_channels_list': [1024, 2048], 'transform_channels': 256, 'query_scales': (1, ), 
    'feats_channels': 512, 'key_pool_scales': (1, 3, 6, 8), 'dropout': 0.1,
}
```

These arguments will be used during instancing the segmentor where the segmentor type is specified in `SEGMENTOR_CFG['type']`.
Now, SSSegmentation supports the following segmentor types,

```python
REGISTERED_MODULES = {
    'FCN': FCN, 'CE2P': CE2P, 'ICNet': ICNet, 'ISNet': ISNet, 'CCNet': CCNet, 'DANet': DANet,
    'GCNet': GCNet, 'DMNet': DMNet, 'ISANet': ISANet, 'ENCNet': ENCNet, 'APCNet': APCNet, 'SAM': SAM,
    'EMANet': EMANet, 'PSPNet': PSPNet, 'PSANet': PSANet, 'OCRNet': OCRNet, 'DNLNet': DNLNet,
    'ANNNet': ANNNet, 'SETRUP': SETRUP, 'SETRMLA': SETRMLA, 'FastFCN': FastFCN, 'UPerNet': UPerNet,
    'Segformer': Segformer, 'MCIBI': MCIBI, 'PointRend': PointRend, 'Deeplabv3': Deeplabv3,
    'LRASPPNet': LRASPPNet, 'MaskFormer': MaskFormer, 'MCIBIPlusPlus': MCIBIPlusPlus, 'SemanticFPN': SemanticFPN,
    'NonLocalNet': NonLocalNet, 'Deeplabv3Plus': Deeplabv3Plus, 'DepthwiseSeparableFCN': DepthwiseSeparableFCN,
    'MobileSAM': MobileSAM, 'IDRNet': IDRNet, 'Mask2Former': Mask2Former,
}
```

To learn more about how to set the specific arguments for each segmentor, you can jump to [`ssseg/modules/models/segmentors` directory](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/segmentors) to check the source codes of each segmentor.

Also, SSSegmentation provides `BaseSegmentor` class to help the users quickly add a new custom segmentor.

Specifically, if the users want to add a new custom segmentor, you should first create a new file in [`ssseg/modules/models/segmentors` directory](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/segmentors), *e.g.*, [`ssseg/modules/models/segmentors/fcn/fcn.py`](https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/segmentors/fcn/fcn.py).

Then, you can define the segmentor in this file by inheriting `BaseSegmentor`, *e.g.*,

```python
from ..base import BaseSegmentor

'''FCN'''
class FCN(BaseSegmentor):
    def __init__(self, cfg, mode):
        super(FCN, self).__init__(cfg, mode)
	'''forward'''
    def forward(self, x, targets=None):
        pass
```

After that, you should add this custom segmentor class in [`ssseg/modules/models/segmentors/builder.py`](https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/segmentors/builder.py) if you want to use it by simply modifying `SEGMENTOR_CFG`.
Of course, you can also register this custom segmentor by the following codes,

```python
from ssseg.modules import SegmentorBuilder

segmentor_builder = SegmentorBuilder()
segmentor_builder.register('FCN', FCN)
```

From this, you can also call `segmentor_builder.build` to build your own defined segmentors as well as the original supported segmentors.

Finally, the users could jump to the [`ssseg/modules/models/segmentors` directory](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/segmentors) in SSSegmentation to read more source codes of the supported segmentors and thus better learn how to customize the segmentors in SSSegmentation.


## Customize Auxiliary Heads

Auxiliary head is the image decoder that transforms the shallow feature maps to a predicted segmentation mask.
It is first introduced in [PSPNet](https://arxiv.org/pdf/1612.01105.pdf), which is used to help segmentation framework training.

#### Auxiliary Head Config Structure

An example of auxiliary head config is as follows,

```python
SEGMENTOR_CFG['auxiliary'] = {'in_channels': 1024, 'out_channels': 512, 'dropout': 0.1}
```

With the arguments in `SEGMENTOR_CFG['auxiliary']`, the `setauxiliarydecoder` function defined in [`ssseg/modules/models/segmentors/base/base.py`](https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/segmentors/base/base.py) will be called to build the auxiliary head.

To disable auxiliary head, you can simply set `SEGMENTOR_CFG['auxiliary']` as `None`.

#### Add New Custom Auxiliary Head

If the users want to add a new custom auxiliary head, you can first define a new segmentor inherited from `BaseSegmentor` class like following, 

```python
from ..base import BaseSegmentor

'''Deeplabv3'''
class Deeplabv3(BaseSegmentor):
    def __init__(self, arg1, arg2):
        pass
    def forward(self, x, targets=None):
        pass
```

After that, you can define the `setauxiliarydecoder` function in this class by yourselves, *e.g.*,

```python
from ..base import BaseSegmentor

'''Deeplabv3'''
class Deeplabv3(BaseSegmentor):
    def __init__(self, arg1, arg2):
        pass
    def forward(self, x, targets=None):
        pass
	def setauxiliarydecoder(self, auxiliary_cfg):
	    pass
```

And the arguments in `SEGMENTOR_CFG['auxiliary']` could be changed according to the new defined `setauxiliarydecoder` function.


## Customize Normalizations

Normalization layer transforms the inputs to have zero mean and unit variance across some specific dimensions.

#### Normalization Config Structure

An example of normalization config is as follows,

```python
SEGMENTOR_CFG['norm_cfg'] = {'type': 'SyncBatchNorm'}
```

where `type` denotes the normalization layer you want to leverage. Now, SSSegmentation supports the following normalization types,

```python
REGISTERED_MODULES = {
    'LayerNorm': nn.LayerNorm, 'LayerNorm2d': LayerNorm2d, 'GroupNorm': nn.GroupNorm, 'LocalResponseNorm': nn.LocalResponseNorm, 
    'BatchNorm1d': nn.BatchNorm1d, 'BatchNorm2d': nn.BatchNorm2d, 'BatchNorm3d': nn.BatchNorm3d, 'SyncBatchNorm': nn.SyncBatchNorm, 
    'InstanceNorm1d': nn.InstanceNorm1d, 'InstanceNorm2d': nn.InstanceNorm2d, 'InstanceNorm3d': nn.InstanceNorm3d, 'GRN': GRN,
}
```

The other arguments in `SEGMENTOR_CFG['norm_cfg']` are set for instancing the corresponding normalization layer.

To learn more about how to set the specific arguments for each normalization layer, you can jump to [`ssseg/modules/models/backbones/bricks/normalization` directory](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/backbones/bricks/normalization) to check the source codes of each normalization layer.

#### Add New Custom Normalization

If the users want to add a new custom normalization layer, you should first create a new file in [`ssseg/modules/models/backbones/bricks/normalization` directory](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/backbones/bricks/normalization), *e.g.*, [`ssseg/modules/models/backbones/bricks/normalization/grn.py`](https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/backbones/bricks/normalization/grn.py).

Then, you can define the normalization layer in this file by yourselves, *e.g.*,

```python
import torch.nn as nn

'''GRN'''
class GRN(nn.Module):
    def __init__(self, arg1, arg2):
        pass
    def forward(self, x):
        pass
```

After that, you should add this custom normalization class in [`ssseg/modules/models/backbones/bricks/normalization/builder.py`](https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/backbones/bricks/normalization/builder.py) if you want to use it by simply modifying `SEGMENTOR_CFG['norm_cfg']` or `SEGMENTOR_CFG['head']['norm_cfg']`.
Of course, you can also register this custom normalization layer by the following codes,

```python
from ssseg.modules import NormalizationBuilder

norm_builder = NormalizationBuilder()
norm_builder.register('GRN', GRN)
```

From this, you can also call `norm_builder.build` to build your own defined normalization layers as well as the original supported normalization layers.

Finally, the users could jump to the [`ssseg/modules/models/backbones/bricks/normalization` directory](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/backbones/bricks/normalization) in SSSegmentation to read more source codes of the supported normalization classes and thus better learn how to customize the normalization layers in SSSegmentation.


## Customize Activations

Activation layer is linear or non linear equation which processes the output of a neuron and bound it into a limited range of values (*e.g.*, $[0, +\infinity]$).

#### Activation Config Structure

An example of activation config is as follows,

```python
SEGMENTOR_CFG['act_cfg'] = {'type': 'ReLU', 'inplace': True}
```

where `type` denotes the activation layer you want to employ. Now, SSSegmentation supports the following activation types,

```python
REGISTERED_MODULES = {
    'ReLU': nn.ReLU, 'GELU': nn.GELU, 'ReLU6': nn.ReLU6, 'PReLU': nn.PReLU,
    'Sigmoid': nn.Sigmoid, 'HardSwish': HardSwish, 'LeakyReLU': nn.LeakyReLU,
    'HardSigmoid': HardSigmoid, 'Swish': Swish,
}
```

The other arguments in `SEGMENTOR_CFG['act_cfg']` are set for instancing the corresponding activation layer.

To learn more about how to set the specific arguments for each activation function, you can jump to [`ssseg/modules/models/backbones/bricks/activation` directory](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/backbones/bricks/activation) to check the source codes of each activation function.

#### Add New Custom Activation

If the users want to add a new custom activation layer, you should first create a new file in [`ssseg/modules/models/backbones/bricks/activation` directory](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/backbones/bricks/activation), *e.g.*, [`ssseg/modules/models/backbones/bricks/activation/hardsigmoid.py`](https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/backbones/bricks/activation/hardsigmoid.py).

Then, you can define the activation layer in this file by yourselves, *e.g.*,

```python
import torch.nn as nn

'''HardSigmoid'''
class HardSigmoid(nn.Module):
    def __init__(self, arg1, arg2):
        pass
    def forward(self, x):
        pass
```

After that, you should add this custom activation class in [`ssseg/modules/models/backbones/bricks/activation/builder.py`](https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/backbones/bricks/activation/builder.py) if you want to use it by simply modifying `SEGMENTOR_CFG['act_cfg']` or `SEGMENTOR_CFG['head']['act_cfg']`.
Of course, you can also register this custom activation layer by the following codes,

```python
from ssseg.modules import ActivationBuilder

act_builder = ActivationBuilder()
act_builder.register('HardSigmoid', HardSigmoid)
```

From this, you can also call `act_builder.build` to build your own defined activation layers as well as the original supported activation layers.

Finally, the users could jump to the [`ssseg/modules/models/backbones/bricks/activation` directory](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/backbones/bricks/activation) in SSSegmentation to read more source codes of the supported activation classes and thus better learn how to customize the activation layers in SSSegmentation.


## Mixed Precision Training

Mixed precision methods combine the use of different numerical formats in one computational workload.
It offers significant computational speedup by performing operations in half-precision format, while storing minimal information in single-precision to retain as much information as possible in critical parts of the network.

SSSegmentation supports two types of mixed precision training, *i.e.*,

- `apex`: Mixed precision training implemented by using the third-party python package `apex` supported by NVIDIA,
- `pytorch`: Mixed precision training implemented by using `torch.cuda.amp` supported by Pytorch official API.

To turn on the mixed precision training in SSSegmentation, you could modify the corresponding config file with the following codes,

```python
import torch

# use Mixed Precision (FP16) Training supported by Apex
SEGMENTOR_CFG['fp16_cfg'] = {'type': 'apex', 'initialize': {'opt_level': 'O1'}, 'scale_loss': {}}
# use Mixed Precision (FP16) Training supported by Pytorch
SEGMENTOR_CFG['fp16_cfg'] = {'type': 'pytorch', 'autocast': {'dtype': torch.float16}, 'grad_scaler': {}}
```

If you choose to use the mixed precision training supported by [Apex](https://nvidia.github.io/apex/), the following arguments could be given,

- `initialize`: Arguments `dict` for instancing `apex.amp.initialize`,
- `scale_loss`: Arguments `dict` for calling `apex.amp.scale_loss`.

The detailed usage and the explanations of each argument please refer to [Apex Official Document](https://nvidia.github.io/apex/).

Of course, you can also choose to adopt the mixed precision training supported by [Pytorch](https://pytorch.org/docs/stable/amp.html#module-torch.amp) and the following arguments could be given,

- `autocast`: Arguments `dict` for instancing `torch.cuda.amp.autocast`,
- `grad_scaler`: Arguments `dict` for instancing `torch.cuda.amp.GradScaler`.

The detailed usage and the explanations of each argument please refer to [Pytorch Official Document](https://pytorch.org/docs/stable/amp.html#module-torch.amp).

Finally, if you want turn off the mixed precision training in SSSegmentation, just delete `fp16_cfg` in `SEGMENTOR_CFG` or set `SEGMENTOR_CFG['fp16_cfg']['type']` as `None`.