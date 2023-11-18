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

If the users want to learn more about this part, it is recommended that you could jump to the <span style="border-bottom:2px dashed blue;">[`datasets directory`](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/datasets)</span> in SSSegmentation to read the source codes of dataset classes.

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

where `Resize` means a data transform method defined in <span style="border-bottom:2px dashed blue;">[`transforms.py`](https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/datasets/pipelines/transforms.py)</span> and other values denote for the arguments for instancing the corresponding data transform method.

Here is a list of supported data transform methods,

```python
REGISTERED_MODULES = {
    'Resize': Resize, 'RandomCrop': RandomCrop, 'RandomFlip': RandomFlip, 'RandomRotation': RandomRotation, 'EdgeExtractor': EdgeExtractor,
    'PhotoMetricDistortion': PhotoMetricDistortion, 'Padding': Padding, 'ToTensor': ToTensor, 'ResizeShortestEdge': ResizeShortestEdge,
    'Normalize': Normalize, 'RandomChoiceResize': RandomChoiceResize, 'Rerange': Rerange, 'CLAHE': CLAHE, 'RandomCutOut': RandomCutOut, 
    'AlbumentationsWrapper': AlbumentationsWrapper, 'RGB2Gray': RGB2Gray, 'AdjustGamma': AdjustGamma,
}
```

To learn the functions of each data transform method, please check the source codes in <span style="border-bottom:2px dashed blue;">[`transforms.py`](https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/datasets/pipelines/transforms.py)</span> by yourselves.

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
from ssseg.modules.datasets import DataTransformBuilder

data_transformer_builder = DataTransformBuilder()
data_transformer_builder.register('RGB2Gray', RGB2Gray)
```

From this, you can call `data_transformer_builder.build` to build your own defined transform algorithms as well as the original supported data transform methods.

#### Add New Custom Dataset

SSSegmentation provide `BaseDataset` class to help the users quickly add a new custom dataset. 

Specifically, you can directly inherit this class to define your own dataset class. Here is an example code to add `SuperviselyDataset` dataset in SSSegmentation <span style="border-bottom:2px dashed blue;">[`datasets directory`](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/datasets)</span> by using `BaseDataset`,

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

Besides, `clsid2label` could be set for transferrin the label ids in ground truth segmentation mask to continuous training ids automatically during training. 
For example, `clsid2label = {10: 1}` means if the class id in ground truth segmentation mask is 10, they will be set as class id 1 before feeding into the models.
Other attributes `classnames` is used to set the class names in the dataset, `palette` is used to set the colors for each class during [Inference with Segmentors Integrated in SSSegmentation](https://sssegmentation.readthedocs.io/en/latest/QuickRun.html#inference-with-segmentors-integrated-in-sssegmentation) and `num_classes` represents the number of the classes existed in the dataset.

After that, you should add this custom dataset class in <span style="border-bottom:2px dashed blue;">[`dataset builder.py`](https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/datasets/builder.py)</span> if you want to use it by modifying `SEGMENTOR_CFG['dataset']`.
Of course, you can also register this custom dataset by the following codes,

```python
from ssseg.modules.datasets import DatasetBuilder

dataset_builder = DatasetBuilder()
dataset_builder.register('SuperviselyDataset', SuperviselyDataset)
```

From this, you can also call `dataset_builder.build` to build your own defined dataset class as well as the original supported data classes.


## Customize Backbones

## Customize Losses

## Customize Optimizers

## Customize Schedulers

## Customize Segmentors

## Mixed Precision Training