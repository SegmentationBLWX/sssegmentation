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

(1) Dataset Configuration

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

(2) Dataloader Configuration

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

(3) Segmentor Configuration

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

Thanks to SSSegmentation’s modular design, switching between datasets requires nothing more than updating the `SEGMENTOR_CFG['dataset']` field, allowing for seamless experimentation across different datasets without modifying the underlying code.

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

The `type` field specifies the dataset class to use. SSSegmentation currently supports the following dataset types,

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
- `data_pipelines`: A list of transformation operations applied sequentially to the input. These transforms are used to preprocess the `sample_meta` objects before they are passed into the model. See [customize-data-pipelines](https://sssegmentation.readthedocs.io/en/latest/Tutorials.html#customize-data-pipelines) for details.

Additional optional fields supported in `SEGMENTOR_CFG['dataset']` include,

- `repeat_times` (*int, default=1*): If set to a value >1, each image will appear multiple times within an epoch, which can be useful for small datasets.
- `eval_env` (*str, default='local'*): Defines the evaluation environment. Options: `local` (evaluate using local ground truth annotations) and `server` (only saves predicted results for submission to external servers).
- `ignore_index` (*int, default=-100*): Label index to ignore during loss computation and evaluation.
- `auto_correct_invalid_seg_target` (*bool, default=False*): If True, automatically fixes invalid pixel values in segmentation targets.

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

where `type` specifies the backbone model to be used. SSSegmentation currently supports the following backbone types:

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

① Add to the Builder File: Modify [`ssseg/modules/models/backbones/builder.py`](https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/backbones/builder.py) to include your new class.

② Register Dynamically: Alternatively, register it manually with the following code,

```python
from ssseg.modules import BackboneBuilder

backbone_builder = BackboneBuilder()
backbone_builder.register('MobileNet', MobileNet)
```

Once registered, you can use `backbone_builder.build(...)` to instantiate either your custom backbone or any of the existing ones.

To gain a deeper understanding, refer to the existing implementation of supported backbones in the [`ssseg/modules/models/backbones`](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/backbones) directory.


## Customize Losses

Loss functions define the optimization objectives for the segmentation framework, for example, the commonly used [Cross Entropy Loss](https://en.wikipedia.org/wiki/Cross-entropy).

#### Loss Config Structure

A typical loss configuration in `SEGMENTOR_CFG` is shown below,

```python
SEGMENTOR_CFG['losses'] = {
    'loss_aux': {'type': 'CrossEntropyLoss', 'scale_factor': 0.4, 'ignore_index': 255, 'reduction': 'mean'},
    'loss_cls': {'type': 'CrossEntropyLoss', 'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'},
}
```

This configuration is a dictionary with keys such as `loss_aux` and `loss_cls`, which distinguish between different loss components, typically used for the auxiliary head and main head respectively.
Each value can be either a dictionary or a list of dictionaries. Within each dictionary, `type` specifies the type of loss function to be used.
SSSegmentation currently supports the following built-in loss types,

```python
REGISTERED_MODULES = {
    'L1Loss': L1Loss, 'MSELoss': MSELoss, 'FocalLoss': FocalLoss, 'CosineSimilarityLoss': CosineSimilarityLoss, 
    'DiceLoss': DiceLoss, 'KLDivLoss': KLDivLoss, 'LovaszLoss': LovaszLoss, 'CrossEntropyLoss': CrossEntropyLoss, 
}
```

The remaining key-value pairs serve as initialization arguments for the corresponding loss function. Commonly used arguments include,

- `scale_factor` (*float, default: 1.0*): A scaling multiplier applied to the computed loss.
- `ignore_index` (*int, default: -100*): Specifies a label value to be ignored during loss computation. For label-based targets, the corresponding pixels will be excluded from the gradient computation. For logit-based targets, the class channel with this index will be ignored.
- `lowest_loss_value` (*float, default: None*): Optionally constrains the loss value with an upper bound. When set, the returned loss becomes `min(lowest_loss_value, scale_factor * original loss)`. This strategy is inspired by [Do We Need Zero Training Loss After Achieving Zero Training Error? - ICML 2020](https://arxiv.org/pdf/2002.08709.pdf).

To support more complex training objectives, each loss component (*e.g.*, `loss_aux` or `loss_cls`) can also be defined as a list of dictionaries, where each dictionary specifies a separate loss term. During training, all specified loss terms will be computed and summed. For example,

```python
loss = 0
for l_cfg in SEGMENTOR_CFG['losses']['loss_aux']:
    loss = loss + BuildLoss(l_cfg)(prediction, target)
```

This design allows for flexible composition of multiple loss functions, enabling finer control over the training dynamics.
For more details on configuring each loss type, refer to the source files in [`ssseg/modules/models/losses`](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/losses) directory.

#### Add New Custom Loss

To integrate a custom loss function into SSSegmentation, follow these steps,

**Step1: Create a New File**

Add a Python file to the [`ssseg/modules/models/losses`](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/losses) directory, for example, [`ssseg/modules/models/losses/kldivloss.py`](https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/losses/kldivloss.py).

**Step2: Define the Loss Class**

Implement your custom loss function. For example,

```python
import torch.nn as nn

'''KLDivLoss'''
class KLDivLoss(nn.Module):
    def __init__(self, arg1, arg2):
        pass
    def forward(self, prediction, target):
        pass
```

**Step3: Register the Loss Function**

You have two options,

① Static Registration. Add the class name to the registration dictionary in [`ssseg/modules/models/losses/builder.py`](https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/losses/builder.py).

② Dynamic Registration. Alternatively, register the class programmatically,

```python
from ssseg.modules import LossBuilder

loss_builder = LossBuilder()
loss_builder.register('KLDivLoss', KLDivLoss)
```

After registration, you can call `loss_builder.build(...)` to instantiate your custom loss, just like with the built-in losses.

To better understand how loss functions are implemented and structured, we recommend reviewing the source code in the [`ssseg/modules/models/losses`](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/losses) directory.


## Customize Schedulers

Schedulers control how the learning rate evolves throughout training, based on either epochs or iterations. 
They are crucial for accelerating convergence and improving training stability.

#### Scheduler Config Structure

A typical learning rate scheduler configuration in `SEGMENTOR_CFG` is as follows,

```python
SEGMENTOR_CFG['scheduler'] = {
    'type': 'PolyScheduler', 'max_epochs': 0, 'power': 0.9,
    'optimizer': {
        'type': 'SGD', 'lr': 0.01, 'momentum': 0.9, 'weight_decay': 5e-4, 'params_rules': {},
    }
}
```

In this configuration, the `type` field specifies the scheduler strategy to be used. SSSegmentation currently supports the following built-in scheduler types,

```python
REGISTERED_MODULES = {
    'PolyScheduler': PolyScheduler, 'CosineScheduler': CosineScheduler,
}
```

The remaining key-value pairs provide initialization arguments for the selected scheduler class. Commonly used arguments include,

- `max_epochs`: The total number of training epochs.
- `power`: The exponent used for polynomial decay (applicable for `PolyScheduler`).
- `optimizer`: A nested dictionary specifying the optimizer settings (see [`customize-optimizers`](https://sssegmentation.readthedocs.io/en/latest/Tutorials.html#customize-optimizers)).

For in-depth information on each scheduler's implementation and additional options, please consult the source code in the [`ssseg/modules/models/schedulers`](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/schedulers) directory.


#### Customize Optimizers

Optimizers play a central role in training by updating model parameters to minimize the loss function. A commonly used optimizer is [Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent), though many other variants are supported.

In SSSegmentation, optimizers are typically defined inside the scheduler configuration block,

```python
SEGMENTOR_CFG['scheduler']['optimizer'] = {
    'type': 'SGD', 'lr': 0.01, 'momentum': 0.9, 'weight_decay': 5e-4, 'params_rules': {},
}
```

where `type` specifies the optimizer to use. SSSegmentation supports a wide range of optimizers, including (**Note**: these optimizers are registered dynamically based on availability in `torch.optim`),

```python
REGISTERED_MODULES = {
    'SGD': optim.SGD, 'Adam': optim.Adam, 'AdamW': optim.AdamW, 'Adadelta': optim.Adadelta,
}
for optim_type in ['Adagrad', 'SparseAdam', 'Adamax', 'ASGD', 'LBFGS', 'NAdam', 'RAdam', 'RMSprop', 'Rprop']:
    if hasattr(optim, optim_type):
        REGISTERED_MODULES[optim_type] = getattr(optim, optim_type)
```

All other key-value pairs are passed as arguments to the optimizer's constructor.
Among these arguments, the `params_rules` field enables fine-grained control over optimization settings for different parts of the model.
This is useful for implementing training strategies such as layer-specific learning rates or selectively disabling weight decay.

(1) Example 1: Assigning a lower learning rate to the backbone,

```python
SEGMENTOR_CFG['scheduler']['optimizer']['params_rules'] = {
    'backbone_net': dict(lr_multiplier=0.1, wd_multiplier=1.0),
}
```

This configuration applies a learning rate that is 0.1× the base rate to all parameters under `backbone_net`, while keeping the weight decay unchanged.

(2) Example 2: Disabling weight decay for specific components

```python
SEGMENTOR_CFG['scheduler']['optimizer']['params_rules'] = {
    'absolute_pos_embed': dict(wd_multiplier=0.),
    'relative_position_bias_table': dict(wd_multiplier=0.),
    'norm': dict(wd_multiplier=0.),
}
```

This is commonly used to prevent regularization on embeddings or normalization layers. 
For more real-world examples of how to leverage `params_rules` for custom optimization strategies, please refer to the [`ssseg/configs`](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/configs) directory.

If you need to implement a custom optimizer for advanced use cases, you can easily extend SSSegmentation by following these steps.

(1) Define Your Optimizer

Create a new Python file in the [`ssseg/modules/models/optimizers`](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/optimizers), *e.g.*, `sgd.py`.
In this file, you can define your custom optimizer class according to your specific requirements,

```python
class SGD():
    def __init__(self, arg1, arg2):
        pass
```

(2) Register the Optimizer

To make your custom optimizer available via `SEGMENTOR_CFG['scheduler']['optimizer']`, you need to register it.
There are two ways to do this,

① Modify the central registry manually: Edit the [`ssseg/modules/models/optimizers/builder.py`](https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/optimizers/builder.py) file and add your custom optimizer to the registration logic.

② Register dynamically in your script or module: Use the `OptimizerBuilder` to register your custom optimizer at runtime,

```python
from ssseg.modules import OptimizerBuilder

optimizer_builder = OptimizerBuilder()
optimizer_builder.register('SGD', SGD)
```

This makes your optimizer accessible via `optimizer_builder.build(...)`. This flexible mechanism allows you to seamlessly integrate your own optimizer while retaining compatibility with the existing configuration system.

#### Add New Custom Scheduler

SSSegmentation provides a base class `BaseScheduler` to help users easily implement and integrate custom learning rate schedulers.

To add a new scheduler,

**Step1: Create a New Scheduler File**

First, create a new Python file in the [`ssseg/modules/models/schedulers`](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/schedulers) directory, *e.g.*, [`ssseg/modules/models/schedulers/polyscheduler.py`](https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/schedulers/polyscheduler.py).

**Step2: Define Your Scheduler Class**

In this file, define your custom scheduler by inheriting from `BaseScheduler`. For example,

```python
from .basescheduler import BaseScheduler

'''PolyScheduler'''
class PolyScheduler(BaseScheduler):
    def __init__(self, arg1, arg2):
        pass
    def updatelr(self):
        pass
```

The `updatelr` method should implement the logic for updating the learning rate at each training step or epoch.

**Step3: Register Your Scheduler**

To make your scheduler configurable via `SEGMENTOR_CFG['scheduler']`, register it in [`ssseg/modules/models/schedulers/builder.py`](https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/schedulers/builder.py). Alternatively, you can register it dynamically using,

```python
from ssseg.modules import SchedulerBuilder

scheduler_builder = SchedulerBuilder()
scheduler_builder.register('PolyScheduler', PolyScheduler)
```

Once registered, your custom scheduler can be instantiated through `scheduler_builder.build(...)`. This allows consistent integration with both built-in and user-defined schedulers.

For further reference, you can explore the existing scheduler implementations in the [`ssseg/modules/models/schedulers`](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/schedulers) directory to better understand the structure and customization practices used in SSSegmentation.


## Customize Segmentors

In SSSegmentation, a segmentor first applies a backbone network to extract multi-level feature maps from the input image, and then uses a decoder head to transform these features into semantic segmentation predictions (*e.g.*, [Deeplabv3](https://arxiv.org/pdf/1706.05587.pdf) and [IDRNet](https://arxiv.org/pdf/2310.10755.pdf)).

A typical segmentor head configuration looks like this,

```python
SEGMENTOR_CFG['head'] = {
    'in_channels_list': [1024, 2048], 'transform_channels': 256, 'query_scales': (1, ), 
    'feats_channels': 512, 'key_pool_scales': (1, 3, 6, 8), 'dropout': 0.1,
}
```

These arguments are used when instantiating the segmentor, whose type is specified in `SEGMENTOR_CFG['type']`. SSSegmentation currently supports the following segmentor types,

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

To explore the full list of supported segmentors and their configuration options, refer to the source code in the [`ssseg/modules/models/segmentors`](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/segmentors) directory.

#### Add New Custom Segmentor

SSSegmentation provides a `BaseSegmentor` class to simplify the process of defining custom segmentors.

**Step1: Create a New Segmentor File**

First, create a new Python file in the [`ssseg/modules/models/segmentors`](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/segmentors) directory, *e.g.*, [`ssseg/modules/models/segmentors/fcn/fcn.py`](https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/segmentors/fcn/fcn.py).

**Step2: Implement Your Segmentor Class**

In the new file, define your custom segmentor by inheriting from `BaseSegmentor`,

```python
import torch.nn as nn
from ..base import BaseSegmentor
from ....utils import SSSegOutputStructure

'''FCN'''
class FCN(BaseSegmentor):
    def __init__(self, cfg, mode):
        super(FCN, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        # build decoder
        convs = []
        for idx in range(head_cfg.get('num_convs', 2)):
            if idx == 0:
                conv = nn.Conv2d(head_cfg['in_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False)
            else:
                conv = nn.Conv2d(head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False)
            norm = BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg)
            act = BuildActivation(act_cfg)
            convs += [conv, norm, act]
        convs.append(nn.Dropout2d(head_cfg['dropout']))
        if head_cfg.get('num_convs', 2) > 0:
            convs.append(nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        else:
            convs.append(nn.Conv2d(head_cfg['in_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        self.decoder = nn.Sequential(*convs)
        # build auxiliary decoder
        self.setauxiliarydecoder(cfg['auxiliary'])
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
    '''forward'''
    def forward(self, data_meta):
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        # feed to backbone network
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to decoder
        seg_logits = self.decoder(backbone_outputs[-1])
        # forward according to the mode
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            loss, losses_log_dict = self.customizepredsandlosses(
                seg_logits=seg_logits, annotations=data_meta.getannotations(), backbone_outputs=backbone_outputs, losses_cfg=self.cfg['losses'], img_size=img_size,
            )
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict) if self.mode == 'TRAIN' else SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict, seg_logits=seg_logits)
        else:
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, seg_logits=seg_logits)
        return ssseg_outputs
```

The `forward` method should implement the model's inference and training behavior.

**Step3: Register Your Segmentor**

To enable configuration-based usage, register your custom segmentor class in [`ssseg/modules/models/segmentors/builder.py`](https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/segmentors/builder.py). Alternatively, you can register it dynamically with,

```python
from ssseg.modules import SegmentorBuilder

segmentor_builder = SegmentorBuilder()
segmentor_builder.register('FCN', FCN)
```

After registration, your custom segmentor can be built using `segmentor_builder.build(...)`. This approach ensures compatibility with both user-defined and built-in segmentor types.

To further understand how to customize segmentors, you are encouraged to review the implementation of existing models in the [`ssseg/modules/models/segmentors`](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/segmentors) directory.


## Customize Auxiliary Heads

The auxiliary head is an additional decoder module that transforms shallow feature maps (typically from an intermediate backbone layer) into a segmentation prediction.
It was first introduced in [PSPNet](https://arxiv.org/pdf/1612.01105.pdf) as an auxiliary supervision branch to help stabilize and improve training in deep segmentation networks.

#### Auxiliary Head Config Structure

A typical auxiliary head configuration is defined in the `SEGMENTOR_CFG['auxiliary']` field. For example,

```python
SEGMENTOR_CFG['auxiliary'] = {'in_channels': 1024, 'out_channels': 512, 'dropout': 0.1}
```

These arguments are passed to the `setauxiliarydecoder` function, which is responsible for building the auxiliary decoder. 
The default implementation of this function is provided in [`ssseg/modules/models/segmentors/base/base.py`](https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/segmentors/base/base.py). 

To disable the auxiliary head, simply set `SEGMENTOR_CFG['auxiliary'] = None`.

#### Add New Custom Auxiliary Head

To implement a custom auxiliary head, you can extend the `BaseSegmentor` class and override the `setauxiliarydecoder` method.

**Step1: Define a Custom Segmentor**

```python
from ..base import BaseSegmentor

'''Deeplabv3'''
class Deeplabv3(BaseSegmentor):
    def __init__(self, arg1, arg2):
        pass
    def forward(self, x, targets=None):
        pass
```

**Step2: Implement a Custom `setauxiliarydecoder` Method**

You can now define your own logic for the auxiliary head by overriding the `setauxiliarydecoder` method,

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

You can modify the contents of `SEGMENTOR_CFG['auxiliary']` to match the argument requirements of your custom `setauxiliarydecoder` method.


## Customize Normalizations

Normalization layers standardize input features by adjusting their distribution to have zero mean and unit variance along specific dimensions. 
This helps stabilize and accelerate training.

#### Normalization Config Structure

A typical normalization configuration is defined via the `SEGMENTOR_CFG['norm_cfg']` field. For example,

```python
SEGMENTOR_CFG['norm_cfg'] = {'type': 'SyncBatchNorm'}
```

Here, the `type` field specifies the normalization method to be used. 
SSSegmentation currently supports the following normalization types,

```python
REGISTERED_MODULES = {
    'LayerNorm': nn.LayerNorm, 'LayerNorm2d': LayerNorm2d, 'GroupNorm': nn.GroupNorm, 'LocalResponseNorm': nn.LocalResponseNorm, 
    'BatchNorm1d': nn.BatchNorm1d, 'BatchNorm2d': nn.BatchNorm2d, 'BatchNorm3d': nn.BatchNorm3d, 'SyncBatchNorm': nn.SyncBatchNorm, 
    'InstanceNorm1d': nn.InstanceNorm1d, 'InstanceNorm2d': nn.InstanceNorm2d, 'InstanceNorm3d': nn.InstanceNorm3d, 'GRN': GRN,
}
```

Any additional arguments defined in `SEGMENTOR_CFG['norm_cfg']` will be passed to the constructor of the selected normalization layer.

To explore implementation details or configuration requirements of individual normalization layers, refer to the source files in the [`ssseg/modules/models/backbones/bricks/normalization`](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/backbones/bricks/normalization) directory.

#### Add New Custom Normalization

To integrate a custom normalization layer, follow these steps,

**Step1: Create a New Module File**

Add a new Python file under the [`ssseg/modules/models/backbones/bricks/normalization`](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/backbones/bricks/normalization) directory, *e.g.*, [`ssseg/modules/models/backbones/bricks/normalization/grn.py`](https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/backbones/bricks/normalization/grn.py).

**Step2: Implement Your Normalization Class**

Define the custom layer in the new file. Example,

```python
import torch.nn as nn

'''GRN'''
class GRN(nn.Module):
    def __init__(self, arg1, arg2):
        pass
    def forward(self, x):
        pass
```

**Register the New Layer**

To use the custom normalization through the config file, register it in [`ssseg/modules/models/backbones/bricks/normalization/builder.py`](https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/backbones/bricks/normalization/builder.py).
Alternatively, you can register it dynamically using,

```python
from ssseg.modules import NormalizationBuilder

norm_builder = NormalizationBuilder()
norm_builder.register('GRN', GRN)
```

You can then use `norm_builder.build(...)` to construct both built-in and custom normalization layers.

Finally, users can explore the [`ssseg/modules/models/backbones/bricks/normalization`](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/backbones/bricks/normalization) directory in SSSegmentation to view the source code of supported normalization layers and learn how to customize their own.


## Customize Activations

Activation layers apply linear or nonlinear transformations to the output of a neuron, typically constraining the values within a specific range (*e.g.*, `[0, +∞]`), and are essential for introducing non-linearity into neural networks.

#### Activation Config Structure

A typical activation layer configuration is defined as,

```python
SEGMENTOR_CFG['act_cfg'] = {'type': 'ReLU', 'inplace': True}
```

where `type` specifies the activation function to use. Currently, SSSegmentation supports the following activation types,

```python
REGISTERED_MODULES = {
    'ReLU': nn.ReLU, 'GELU': nn.GELU, 'ReLU6': nn.ReLU6, 'PReLU': nn.PReLU,
    'Sigmoid': nn.Sigmoid, 'HardSwish': HardSwish, 'LeakyReLU': nn.LeakyReLU,
    'HardSigmoid': HardSigmoid, 'Swish': Swish,
}
```

Additional arguments in `SEGMENTOR_CFG['act_cfg']` can be used to configure the selected activation layer.

For details on available arguments and implementations, refer to the [`ssseg/modules/models/backbones/bricks/activation`](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/backbones/bricks/activation) directory in the source code.

#### Add New Custom Activation

To implement a new custom activation function, follow these steps,

**Step1: Create a New Module File**

Add a new Python file in the [`ssseg/modules/models/backbones/bricks/activation`](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/backbones/bricks/activation) directory, *e.g.*, [`ssseg/modules/models/backbones/bricks/activation/hardsigmoid.py`](https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/backbones/bricks/activation/hardsigmoid.py).

**Step2: Define the Activation Class**

Implement the activation function in the new file. Example,

```python
import torch.nn as nn

'''HardSigmoid'''
class HardSigmoid(nn.Module):
    def __init__(self, arg1, arg2):
        pass
    def forward(self, x):
        pass
```

**Step3: Register the Custom Activation**

To make the custom activation available via config files, add it to [`ssseg/modules/models/backbones/bricks/activation/builder.py`](https://github.com/SegmentationBLWX/sssegmentation/blob/main/ssseg/modules/models/backbones/bricks/activation/builder.py). Alternatively, register it manually with,

```python
from ssseg.modules import ActivationBuilder

act_builder = ActivationBuilder()
act_builder.register('HardSigmoid', HardSigmoid)
```

You can then use `act_builder.build(...)` to construct both standard and custom activation layers.

For more examples and implementation details, explore the [`ssseg/modules/models/backbones/bricks/activation`](https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/backbones/bricks/activation) directory in the SSSegmentation codebase.


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

```python
SEGMENTOR_CFG['ema_cfg'] = {'momentum': 0.0005, 'device': 'cpu'}
```

where `device` denotes perform EMA on CPU or GPU, `momentum` is the moving average weight which is used in the following codes,

```python
ema_v * (1.0 - momentum)) + (momentum * cur_v)
```

Finally, if you want turn off EMA in SSSegmentation, just delete `ema_cfg` in `SEGMENTOR_CFG` or set `SEGMENTOR_CFG['ema_cfg']['momentum']` as `None`.
