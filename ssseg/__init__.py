'''version'''
__version__ = '1.5.3'
'''author'''
__author__ = 'Zhenchao Jin'
'''title'''
__title__ = 'SSSegmentation'
'''description'''
__description__ = 'SSSegmentation: An Open Source Supervised Semantic Segmentation Toolbox Based on PyTorch'
'''url'''
__url__ = 'https://github.com/SegmentationBLWX/sssegmentation'
'''email'''
__email__ = 'charlesblwx@gmail.com'
'''license'''
__license__ = 'Apache License 2.0'
'''copyright'''
__copyright__ = 'Copyright 2020-2030 Zhenchao Jin'


'''import all'''
from . import configs
from .test import Tester
from .train import Trainer
from .inference import Inferencer
from .modules import (
    DatasetBuilder, DataTransformBuilder, BuildDataset, BuildDataTransform, BuildDistributedDataloader, BuildDistributedModel,
    LossBuilder, BackboneBuilder, SegmentorBuilder, PixelSamplerBuilder, OptimizerBuilder, SchedulerBuilder, ParamsConstructorBuilder, 
    NormalizationBuilder, ActivationBuilder, DropoutBuilder, BuildLoss, BuildBackbone, BuildSegmentor, BuildPixelSampler, BuildOptimizer, 
    BuildScheduler, BuildParamsConstructor, BuildNormalization, BuildActivation, BuildDropout, EMASegmentor, EnvironmentCollector,
    initslurm, setrandomseed, touchdir, loadckpts, saveckpts, loadpretrainedweights, symlink, judgefileexist, postprocesspredgtpairs,
    LoggerHandleBuilder, BuildLoggerHandle, TrainingLoggingManager, BaseModuleBuilder, ConfigParser
)