# Tutorials

In this chapter, we will provide detailed tutorials to help the users learn how to use SSSegmentation.


## Learn about Config

We incorporate modular design into our config system, which is convenient to conduct various experiments. 

#### Config File Structure

Now, there are two basic component types under `ssseg/configs/_base_`, *i.e.*, `datasets` and `dataloaders`, which define configurations for various datasets under different runtime settings (*e.g.*, batch size, image size, data augmentation, to name a few).

For example, to train an FCN segmentor on the Pascal VOC dataset (assuming a total batch size of 16 and an image size of 512×512), you can import the corresponding pre-defined configs like this,

```python
from .._base_ import REGISTERED_SEGMENTOR_CONFIGS, REGISTERED_DATASET_CONFIGS, REGISTERED_DATALOADER_CONFIGS
```

Then, modify `SEGMENTOR_CFG` in the method-specific config file (*e.g.*, `ssseg/configs/fcn/fcn_resnet50os16_voc.py`) as follows,

```python
# deepcopy
SEGMENTOR_CFG = REGISTERED_SEGMENTOR_CONFIGS['FCN_SEGMENTOR_CFG'].copy()
# modify dataset config
SEGMENTOR_CFG['dataset'] = REGISTERED_DATASET_CONFIGS['DATASET_CFG_VOCAUG_512x512'].copy()
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = REGISTERED_DATALOADER_CONFIGS['DATALOADER_CFG_BS16'].copy()
```

With this modular structure, you no longer need to repeatedly define dataloader and dataset configs when using standard settings across segmentation tasks.

Next, let’s discuss config files in specific segmentation algorithm directories (*e.g.*, `ssseg/configs/fcn`). 
Previously (`SSSegmentation <= 1.6.0`), each method folder contained a `base_cfg.py` file to define the core configuration for that algorithm. 
Now, these base configs have been moved to `ssseg/configs/_base_/segmentors/` and renamed according to the algorithm name (*e.g.*, `fcn.py` for FCN, `deeplabv3.py` for DeepLabV3).
You can import the corresponding base segmentor config like this:

```
# way1
from .._base_ import REGISTERED_SEGMENTOR_CONFIGS
SEGMENTOR_CFG = REGISTERED_SEGMENTOR_CONFIGS['FCN_SEGMENTOR_CFG'].copy()
# way2
from .._base_.segmentors.fcn import FCN_SEGMENTOR_CFG as SEGMENTOR_CFG
```

Then, customize and train the model by modifying key fields. For instance, to use FCN with a ResNet-50-D16 backbone on Pascal VOC, you can create a config file named `fcn_resnet50os16_voc.py` under `ssseg/configs/fcn`,

```python
import os
from .._base_ import REGISTERED_SEGMENTOR_CONFIGS, REGISTERED_DATASET_CONFIGS, REGISTERED_DATALOADER_CONFIGS


# deepcopy
SEGMENTOR_CFG = REGISTERED_SEGMENTOR_CONFIGS['FCN_SEGMENTOR_CFG'].copy()
# modify dataset config
SEGMENTOR_CFG['dataset'] = REGISTERED_DATASET_CONFIGS['DATASET_CFG_VOCAUG_512x512'].copy()
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = REGISTERED_DATALOADER_CONFIGS['DATALOADER_CFG_BS16'].copy()
# modify scheduler config
SEGMENTOR_CFG['scheduler']['max_epochs'] = 60
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 21
SEGMENTOR_CFG['backbone'] = {
    'type': 'ResNet', 'depth': 50, 'structure_type': 'resnet50conv3x3stem',
    'pretrained': True, 'outstride': 16, 'use_conv3x3_stem': True, 'selected_indices': (2, 3),
}
SEGMENTOR_CFG['work_dir'] = os.path.split(__file__)[-1].split('.')[0]
SEGMENTOR_CFG['logger_handle_cfg']['logfilepath'] = os.path.join(SEGMENTOR_CFG['work_dir'], f"{os.path.split(__file__)[-1].split('.')[0]}.log")
```

To start training this model, run,

```sh
bash scripts/dist_train.sh 4 ssseg/configs/fcn/fcn_resnet50os16_voc.py
```

How relaxing and enjoyable! You can explore more configuration examples under `ssseg/configs` to better understand how to define valid config files in SSSegmentation.

#### An Example of PSPNet

To help users understand the structure of a complete config and the modular components in SSSegmentation, 
we provide a commented example using PSPNet with ResNet-101-D8, reflecting the new configuration system based on `SegmentorConfig`, `DatasetConfig` and `DataloaderConfig`.

**Dataset Configuration**,

```python
DATASET_CFG_ADE20k_512x512 = DatasetConfig(
    type='ADE20kDataset',  # dataset type; used to instantiate dataset class in builder.py
    rootdir=os.path.join(os.getcwd(), 'ADE20k'),  # path to dataset directory
    train={
        'set': 'train',  # split used for training
        'data_pipelines': [
            ('Resize', {'output_size': (2048, 512), 'keep_ratio': True, 'scale_range': (0.5, 2.0)}),
            ('RandomCrop', {'crop_size': (512, 512), 'one_category_max_ratio': 0.75}),
            ('RandomFlip', {'prob': 0.5}),
            ('PhotoMetricDistortion', {}),
            ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
            ('ToTensor', {}),
            ('Padding', {'output_size': (512, 512), 'data_type': 'tensor'}),
        ]  # data preprocessing pipeline for training; see ssseg/modules/datasets/pipelines
    },
    test={
        'set': 'val',
        'data_pipelines': [
            ('Resize', {'output_size': (2048, 512), 'keep_ratio': True, 'scale_range': None}),
            ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
            ('ToTensor', {}),
        ]  # data preprocessing pipeline for validation
    }
)
```

**Dataloader Configuration**,

```python
DATALOADER_CFG_BS16 = DataloaderConfig(
    # Expected total training batch size across all GPUs. This is used to verify whether users set a reasonable batch size and detect misconfiguration.
    expected_total_train_bs_for_assert=16,
    # If True, the framework will automatically calculate the per-GPU batch size based on the number of available GPUs and `expected_total_train_bs_for_assert`.
    # If False, the values in `train['batch_size_per_gpu']` will be used directly.
    auto_adapt_to_expected_train_bs=True,
    train={
        'batch_size_per_gpu': 2,   # Number of training samples per GPU (used only when `auto_adapt_to_expected_train_bs` is False).
        'num_workers_per_gpu': 2,  # Number of worker processes per GPU for loading data in parallel.
        'shuffle': True,           # Whether to shuffle the training data at each epoch.
        'pin_memory': True,        # If True, the data loader will copy tensors into CUDA pinned memory before returning them. This can accelerate host-to-device transfers.
        'drop_last': True,         # If True, drops the last incomplete batch during training if the dataset size is not divisible by the batch size. Useful to maintain consistent batch sizes.
    },
    test={
        'batch_size_per_gpu': 1,  # Number of validation/test samples per GPU, only support setting as 1 now.
        'num_workers_per_gpu': 2, # Number of worker processes per GPU for loading validation/test data.
        'shuffle': False,         # Whether to shuffle the test data. Usually set to False for reproducibility.
        'pin_memory': True,       # Whether to enable pinned memory for test data loading.
        'drop_last': False,       # Whether to drop the last incomplete test batch. Usually set to False to ensure full evaluation.
    }
)
```

**Segmentor Configuration**,

```python
PSPNET_SEGMENTOR_CFG = SegmentorConfig(
    type='PSPNet',  # segmentor type defined in ssseg/modules/models/segmentors
    num_classes=-1,  # number of output classes (to be set later)
    benchmark=True,  # enables cudnn.benchmark for performance
    align_corners=False,  # used in torch.nn.functional.interpolate
    work_dir='ckpts',  # directory for logs and checkpoints
    eval_interval_epochs=10,  # evaluation frequency
    save_interval_epochs=1,  # checkpoint saving frequency
    logger_handle_cfg={'type': 'LocalLoggerHandle', 'logfilepath': ''},
    training_logging_manager_cfg={'log_interval_iters': 50},
    norm_cfg={'type': 'SyncBatchNorm'},  # normalization config
    act_cfg={'type': 'ReLU', 'inplace': True},  # activation config
    # backbone config (ResNet-101 with 3x3 stem)
    backbone={
        'type': 'ResNet', 'depth': 101, 'structure_type': 'resnet101conv3x3stem',
        'pretrained': True, 'outstride': 8, 'use_conv3x3_stem': True, 'selected_indices': (2, 3),
    },
    # PSPNet head
    head={
        'in_channels': 2048, 'feats_channels': 512, 'pool_scales': [1, 2, 3, 6], 'dropout': 0.1,
    },
    # auxiliary decoder head
    auxiliary={
        'in_channels': 1024, 'out_channels': 512, 'dropout': 0.1,
    },
    # loss functions
    losses={
        'loss_aux': {'type': 'CrossEntropyLoss', 'scale_factor': 0.4, 'ignore_index': -100, 'reduction': 'mean'},
        'loss_cls': {'type': 'CrossEntropyLoss', 'scale_factor': 1.0, 'ignore_index': -100, 'reduction': 'mean'},
    },
    # inference settings
    inference={
        'forward': {'mode': 'whole', 'cropsize': None, 'stride': None},
        'tta': {'multiscale': [1], 'flip': False, 'use_probs_before_resize': False},
        'evaluate': {'metric_list': ['iou', 'miou']},
    },
    # scheduler and optimizer
    scheduler={
        'type': 'PolyScheduler', 'max_epochs': 0, 'power': 0.9,
        'optimizer': {
            'type': 'SGD', 'lr': 0.01, 'momentum': 0.9, 'weight_decay': 5e-4, 'params_rules': {},
        }
    },
    # dataset and dataloader (to be assigned later)
    dataset=None,
    dataloader=None,
)
```

In the next sections, we’ll dive deeper into each module and its options to help you fully customize your segmentation pipeline.


## Customize Datasets

In SSSegmentation, dataset classes serve two primary purposes,

- Loading dataset information after data preparation.
- Constructing `sample_meta` dictionaries that encapsulate all necessary metadata required for training and testing the segmentor.

Each `sample_meta` is a `dict` containing the following keys,

- `image`: The loaded input image tensor.
- `seg_target`: The corresponding ground truth segmentation mask.
- `edge_target` (*optional*): The edge mask derived from the segmentation mask.
- `img2aug_pos_mapper` (*optional*): A pixel-wise mapping from the original image to its augmented counterpart.
- `width` and `height`: The original dimensions of the image (*i.e.*, before any data augmentation or resizing).
- `id`: A unique identifier for the image.

Thanks to SSSegmentation’s modular design, switching datasets is as simple as updating the `SEGMENTOR_CFG['dataset']` field—enabling flexible experimentation across different datasets without code changes.

#### Dataset Config Structure

The dataset configuration in SSSegmentation is defined using the `DatasetConfig` class. Below is an example for configuring the ADE20k dataset with 512×512 input size,

```python
import os
from .default_dataset import DatasetConfig


'''DATASET_CFG_ADE20k_512x512'''
DATASET_CFG_ADE20k_512x512 = DatasetConfig(
    type='ADE20kDataset',
    rootdir=os.path.join(os.getcwd(), 'ADE20k'),
    train={
        'set': 'train',
        'data_pipelines': [
            ('Resize', {'output_size': (2048, 512), 'keep_ratio': True, 'scale_range': (0.5, 2.0)}),
            ('RandomCrop', {'crop_size': (512, 512), 'one_category_max_ratio': 0.75}),
            ('RandomFlip', {'prob': 0.5}),
            ('PhotoMetricDistortion', {}),
            ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
            ('ToTensor', {}),
            ('Padding', {'output_size': (512, 512), 'data_type': 'tensor'}),
        ]
    },
    test={
        'set': 'val',
        'data_pipelines': [
            ('Resize', {'output_size': (2048, 512), 'keep_ratio': True, 'scale_range': None}),
            ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
            ('ToTensor', {}),
        ]
    }
)
```

The `type`field specifies the dataset class to use. SSSegmentation currently supports the following dataset types,

```python
REGISTERED_MODULES = {
    'BaseDataset': BaseDataset, 'VOCDataset': VOCDataset, 'PascalContext59Dataset': PascalContext59Dataset, 'PascalContextDataset': PascalContextDataset,
    'COCOVOCSUBDataset': COCOVOCSUBDataset, 'COCOStuff10kDataset': COCOStuff10kDataset, 'COCOStuffDataset': COCOStuffDataset, 'CIHPDataset': CIHPDataset,
    'LIPDataset': LIPDataset, 'ATRDataset': ATRDataset, 'MHPv1Dataset': MHPv1Dataset, 'MHPv2Dataset': MHPv2Dataset, 'SuperviselyDataset': SuperviselyDataset,
    'HRFDataset': HRFDataset, 'ChaseDB1Dataset': ChaseDB1Dataset, 'STAREDataset': STAREDataset, 'DRIVEDataset': DRIVEDataset, 'SBUShadowDataset': SBUShadowDataset,
    'VSPWDataset': VSPWDataset, 'ADE20kDataset': ADE20kDataset, 'DarkZurichDataset': DarkZurichDataset, 'NighttimeDrivingDataset': NighttimeDrivingDataset,
    'CityScapesDataset': CityScapesDataset, 'MultipleDataset': MultipleDataset,
}
```

**Tip:** It is recommended to symlink the dataset folder to the `$SSSEGMENTATION/` root or simply run `bash scripts/prepare_datasets.sh $DATASETNAME` in `$SSSEGMENTATION/`. This way, the default `rootdir` does not need to be manually modified in most cases.

The `train` and `test` fields define the configuration for training and evaluation splits, respectively. Both are dictionaries with the following keys,

- `set`: Indicates which subset of the data to use (*e.g.*, `train`, `val`, `test`).
- `data_pipelines`: A list of transformation operations applied sequentially to the input. These transforms are used to preprocess the `sample_meta` objects before they are passed into the model. See [Customize Data Pipelines](https://sssegmentation.readthedocs.io/en/latest/Tutorials.html#customize-data-pipelines) for details.

Additional optional fields supported in `SEGMENTOR_CFG['dataset']` include,

- `repeat_times` (*int, default=1*): If set to a value >1, each image will appear multiple times within an epoch, which can be useful for small datasets.
- `eval_env` (*str, default='local'*): Defines the evaluation environment. Options: `local` (evaluate using local ground truth annotations) and `server` (only saves predicted results for submission to external servers).
- `ignore_index` (*int, default=-100*): Label index to ignore during loss computation and evaluation.
- `auto_correct_invalid_seg_target (*bool, default=False*)`: If True, automatically fixes invalid pixel values in segmentation targets.

For a deeper understanding, users are encouraged to explore the [`ssseg/modules/datasets`](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/datasets) directory, where the dataset class definitions and data loading logic are implemented.

#### Customize Data Pipelines

In SSSegmentation, data pipelines are used to preprocess input samples such as images and segmentation masks before feeding them into the segmentor for training or testing. 
These pipelines are defined in the dataset configuration under,

- `SEGMENTOR_CFG['dataset']['train']['data_pipelines']`: Data transformations applied during training.
- `SEGMENTOR_CFG['dataset']['test']['data_pipelines']`: Data transformations applied during testing.

Each `data_pipelines` entry is a `list` of operations, with each operation represented as either a `tuple` or `dict`. Below is an example,

```python
SEGMENTOR_CFG['dataset']['train']['data_pipelines'] = [
    ('Resize', {'output_size': (2048, 512), 'keep_ratio': True, 'scale_range': (0.5, 2.0)}),
    ('RandomCrop', {'crop_size': (512, 512), 'one_category_max_ratio': 0.75}),
    ('RandomFlip', {'prob': 0.5}),
    ('PhotoMetricDistortion', {}),
    ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
    ('ToTensor', {}),
    ('Padding', {'output_size': (512, 512), 'data_type': 'tensor'}),
]
```

Each operation can be expressed as,

```
# tuple
('Resize', {'output_size': (2048, 512), 'keep_ratio': True, 'scale_range': (0.5, 2.0)})
# dict
{'type': 'Resize', 'output_size': (2048, 512), 'keep_ratio': True, 'scale_range': (0.5, 2.0)}
```

Here, `Resize` refers to a transformation method implemented in [`ssseg/modules/datasets/pipelines/transforms.py`](https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/datasets/pipelines/transforms.py), 
while the second element or the associated keys specify its initialization arguments.

The following data transformation methods are currently registered,

```python
REGISTERED_MODULES = {
    'Resize': Resize, 'RandomCrop': RandomCrop, 'RandomFlip': RandomFlip, 'RandomRotation': RandomRotation, 'EdgeExtractor': EdgeExtractor,
    'PhotoMetricDistortion': PhotoMetricDistortion, 'Padding': Padding, 'ToTensor': ToTensor, 'ResizeShortestEdge': ResizeShortestEdge,
    'Normalize': Normalize, 'RandomChoiceResize': RandomChoiceResize, 'Rerange': Rerange, 'CLAHE': CLAHE, 'RandomCutOut': RandomCutOut, 
    'AlbumentationsWrapper': AlbumentationsWrapper, 'RGB2Gray': RGB2Gray, 'AdjustGamma': AdjustGamma, 'RandomGaussianBlur': RandomGaussianBlur,
    'RandomShortestEdgeResize': RandomShortestEdgeResize, 'PILRandomGaussianBlur': PILRandomGaussianBlur,
}
```

You can refer to the source code of each method in [`ssseg/modules/datasets/pipelines/transforms.py`](https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/datasets/pipelines/transforms.py) to understand their functionality and configuration options.

SSSegmentation also supports `AlbumentationsWrapper`, which allows you to integrate any [Albumentations](https://albumentations.ai/docs/) transformation within your pipeline. Here’s an example,

```python
SEGMENTOR_CFG['dataset']['train']['data_pipelines'] = [
    ('AlbumentationsWrapper', {'albu_cfg': {'type': 'RandomCrop', 'width': 256, 'height': 256}}),
    ('AlbumentationsWrapper', {'albu_cfg': {'type': 'HorizontalFlip', 'p': 0.5}}),
    ('AlbumentationsWrapper', {'albu_cfg': {'type': 'RandomBrightnessContrast', 'p': 0.2}}),
]
```

Each `albu_cfg` specifies an Albumentations transformation using its native API parameters.

You can also define your own transformation methods. For example,

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

To register this custom transformation,

```python
from ssseg.modules import DataTransformBuilder

data_transformer_builder = DataTransformBuilder()
data_transformer_builder.register('RGB2Gray', RGB2Gray)
```

Once registered, your custom transform can be used in the same way as built-in transformations, via `data_transformer_builder.build(...)`.
This modular and extensible design enables users to flexibly construct and experiment with various data pipelines tailored to different datasets and model requirements.

#### Add New Custom Dataset

SSSegmentation provides a flexible base class, `BaseDataset`, to help users quickly integrate their own custom datasets.

To add a new dataset, simply subclass `BaseDataset` and implement the dataset-specific logic. 
Below is an example of how to add a `SuperviselyDataset` class under the [`ssseg/modules/datasets`](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/datasets) directory.

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

When implementing the `__init__` method, the following attributes should be defined,

- `image_dir`: Path to the directory containing input images.
- `ann_dir`: Path to the directory containing ground truth segmentation masks.
- `image_ext` (*str, default='.jpg'*): File extension for annotations.
- `ann_ext` (*str, default='.png'*): File extension for images.
- `imageids`: A list of image IDs or filenames (without extension).

In addition, the following class-level attributes can be customized,

- `classnames`: A list of semantic class names.
- `clsid2label` (*optional*): A dictionary for remapping original label IDs in the segmentation masks to training IDs. For example, `clsid2label = {10: 1}` means that pixels with value 10 in the original mask will be mapped to class ID 1 before training.
- `palette`: A list of RGB tuples defining the visualization color for each class.
- `num_classes`: The total number of semantic classes in the dataset.

To make the custom dataset usable within the configuration system, you have two options.

(1) Register it manually via code, 

```python
from ssseg.modules import DatasetBuilder

dataset_builder = DatasetBuilder()
dataset_builder.register('SuperviselyDataset', SuperviselyDataset)
```

This allows you to call `dataset_builder.build(...)` to instantiate your custom dataset just like the built-in ones.

(2) **(Recommended)** Add it to the dataset registry in [`ssseg/modules/datasets/builder.py`](https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/datasets/builder.py), so it can be automatically referenced using `SEGMENTOR_CFG['dataset']['type']`.

For more examples and deeper insight into how datasets are handled internally, you are encouraged to explore the [`ssseg/modules/datasets`](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/datasets) directory. 
Studying the existing dataset classes will help you effectively customize and extend the dataset handling in SSSegmentation for your specific use case.


## Customize Backbones

A backbone serves as the image encoder that transforms an input image into feature maps. For example, a typical backbone could be a ResNet-50 without its final fully connected layer.

#### Backbone Config Structure

A typical backbone configuration in `SEGMENTOR_CFG` is shown below:

```python
SEGMENTOR_CFG['backbone'] = {
    'type': 'ResNet', 'depth': 101, 'structure_type': 'resnet101conv3x3stem',
    'pretrained': True, 'outstride': 16, 'use_conv3x3_stem': True, 'selected_indices': (2, 3),
}
```

where,

- `type`: Specifies the backbone model to use.
- `depth`: Indicates the depth of the network (if applicable).
- Other fields are used to configure the behavior of the selected backbone.

SSSegmentation currently supports the following backbone types:

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

Additional configuration parameters in `SEGMENTOR_CFG['backbone']` vary depending on the specific backbone. Some commonly used parameters include:

- `structure_type`: Defines the structure variant of the backbone (*e.g.*, `resnet101conv3x3stem` indicates a ResNet-101 with three 3×3 convolutions in the stem layer). This helps load corresponding pretrained weights automatically.
- `pretrained`: Whether to load pretrained weights.
- `pretrained_model_path`: If set to `None` and `pretrained=True`, the pretrained weights will be loaded automatically. Otherwise, weights are loaded from the specified path.
- `out_indices`: Specifies which stages of the backbone to output. Most backbones are divided into stages, and this parameter selects which stages' outputs are used.
- `norm_cfg`: Dictionary defining the normalization layer. See [customize-normalizations](https://sssegmentation.readthedocs.io/en/latest/Tutorials.html#customize-normalizations) for details.
- `act_cfg`: Dictionary defining the activation function. See [customize-activations](https://sssegmentation.readthedocs.io/en/latest/Tutorials.html#customize-activations) for details.

To understand how to set arguments for each backbone type, refer to the source code in the [`ssseg/modules/models/backbones`](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/backbones) directory.

#### Add New Custom Backbone

To add your own custom backbone, follow these steps:

**Step1: Create a New File**

Add a new Python file under [`ssseg/modules/models/backbones`](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/backbones), such as [`ssseg/modules/models/backbones/mobilenet.py`](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/backbones/mobilenet.py).

**Step2: Define the Backbone Module**

Implement your custom backbone class in the file. For example,

```python
import torch.nn as nn

'''MobileNet'''
class MobileNet(nn.Module):
    def __init__(self, arg1, arg2):
        pass
    def forward(self, x):
        pass
```

**Step3: Register the Custom Backbone**

You can register the new backbone in two ways.

- Add to the Builder File: Modify [`ssseg/modules/models/backbones/builder.py`](https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/backbones/builder.py) to include your new class.
- Register Dynamically: Alternatively, register it manually with the following code,

```python
from ssseg.modules import BackboneBuilder

backbone_builder = BackboneBuilder()
backbone_builder.register('MobileNet', MobileNet)
```

Once registered, you can use `backbone_builder.build(...)` to instantiate either your custom backbone or any of the existing ones.

To gain a deeper understanding, refer to the existing implementation of supported backbones in the [`ssseg/modules/models/backbones`](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/backbones) directory.


## Customize Losses

Loss is utilized to define the objective functions for the segmentation framework, such as the [Cross Entropy Loss](https://en.wikipedia.org/wiki/Cross-entropy).

#### Loss Config Structure

An example of loss config is as follows,

```python
SEGMENTOR_CFG['losses'] = {
    'loss_aux': {'type': 'CrossEntropyLoss', 'scale_factor': 0.4, 'ignore_index': 255, 'reduction': 'mean'},
    'loss_cls': {'type': 'CrossEntropyLoss', 'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'},
}
```

It is a `dict` including several keys like `loss_aux` and `loss_cls`, which is designed for distinguishing the loss config utilized in head or auxiliary head.
The value corresponding to each key is also a `dict` or a list of `dict` and in each `dict`, `type` denotes the objective function type you want to adopt and the other arguments are set for instancing this objective function.

In the example of loss config above, `SEGMENTOR_CFG['losses']['loss_cls']` is used to build losses in head and `SEGMENTOR_CFG['losses']['loss_aux']` is used to build losses in auxiliary head.
`CrossEntropyLoss` represents the objective function type you want to adopt and `{'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'}` contains the arguments for instancing the `CrossEntropyLoss`.

And if `SEGMENTOR_CFG['losses']['loss_aux']` or `SEGMENTOR_CFG['losses']['loss_cls']` is a list of `dict`, the final loss will be calculated as,

```python
loss = 0
for l_cfg in SEGMENTOR_CFG['losses']['loss_aux']:
    loss = loss + BuildLoss(l_cfg)(prediction, target)
```

Now, SSSegmentation supports the following loss types,

```python
REGISTERED_MODULES = {
    'L1Loss': L1Loss, 'MSELoss': MSELoss, 'FocalLoss': FocalLoss, 'CosineSimilarityLoss': CosineSimilarityLoss, 
    'DiceLoss': DiceLoss, 'KLDivLoss': KLDivLoss, 'LovaszLoss': LovaszLoss, 'CrossEntropyLoss': CrossEntropyLoss, 
}
```

Here we also list some common arguments and their explanation,

- `scale_factor`: The return loss value is equal to the original loss times `scale_factor`,
- `ignore_index`: For input target with labels, it is used to specify a target value that is ignored and does not contribute to the input gradient, for input target with logits, it is used to specify a class channel that is ignored and does not contribute to the input gradient,
- `lowest_loss_value`: If `lowest_loss_value` is set, the return loss value is equal to `min(lowest_loss_value, scale_factor * original loss)`, this argument is designed according to the paper [Do We Need Zero Training Loss After Achieving Zero Training Error? - ICML 2020](https://arxiv.org/pdf/2002.08709.pdf).

To learn more about how to set the specific arguments for each loss function, you can jump to [`ssseg/modules/models/losses` directory](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/losses) to check the source codes of each loss function.

#### Add New Custom Loss

If the users want to add a new custom loss, you should first create a new file in [`ssseg/modules/models/losses` directory](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/losses), *e.g.*, [`ssseg/modules/models/losses/kldivloss.py`](https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/losses/kldivloss.py).

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
    'PolyScheduler': PolyScheduler, 'CosineScheduler': CosineScheduler,
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

After that, you should add this custom normalization class in [`ssseg/modules/models/backbones/bricks/normalization/builder.py`](https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/backbones/bricks/normalization/builder.py) if you want to use it by simply modifying `SEGMENTOR_CFG['norm_cfg']` or `SEGMENTOR_CFG['backbone']['norm_cfg']`.
Of course, you can also register this custom normalization layer by the following codes,

```python
from ssseg.modules import NormalizationBuilder

norm_builder = NormalizationBuilder()
norm_builder.register('GRN', GRN)
```

From this, you can also call `norm_builder.build` to build your own defined normalization layers as well as the original supported normalization layers.

Finally, the users could jump to the [`ssseg/modules/models/backbones/bricks/normalization` directory](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/backbones/bricks/normalization) in SSSegmentation to read more source codes of the supported normalization classes and thus better learn how to customize the normalization layers in SSSegmentation.


## Customize Activations

Activation layer is linear or non linear equation which processes the output of a neuron and bound it into a limited range of values (*e.g.*, `[0, +∞]`).

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

After that, you should add this custom activation class in [`ssseg/modules/models/backbones/bricks/activation/builder.py`](https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/backbones/bricks/activation/builder.py) if you want to use it by simply modifying `SEGMENTOR_CFG['act_cfg']` or `SEGMENTOR_CFG['backbone']['act_cfg']`.
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

For more technical details, please refer to [Mixed Precision Training](https://arxiv.org/abs/1710.03740).

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


## Exponential Moving Average (EMA)

Exponential moving average is a neural network training trick that sometimes improves the model accuracy. 
Concretely, instead of using the optimized parameters from the final training iteration (parameter update step) as the final parameters for the model, the exponential moving average of the parameters over the course of all the training iterations are used.

When training a model, it is often beneficial to maintain moving averages of the trained parameters. Evaluations that use averaged parameters sometimes produce significantly better results than the final trained values.

You can refer to [Exponential-Moving-Average](https://leimao.github.io/blog/Exponential-Moving-Average/) for more technical details.

To turn on EMA in SSSegmentation, you could modify the corresponding config file with the following codes,

```
SEGMENTOR_CFG['ema_cfg'] = {'momentum': 0.0005, 'device': 'cpu'}
```

where `device` denotes perform EMA on CPU or GPU, `momentum` is the moving average weight which is used in the following codes,

```
ema_v * (1.0 - momentum)) + (momentum * cur_v)
```

Finally, if you want turn off EMA in SSSegmentation, just delete `ema_cfg` in `SEGMENTOR_CFG` or set `SEGMENTOR_CFG['ema_cfg']['momentum']` as `None`.
