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


#### An Example of PSPNet
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
    'logfilepath': '', # file path to store the training and testing logs
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

