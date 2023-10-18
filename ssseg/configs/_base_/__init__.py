'''initialize'''
from .dataloaders import (
    DATALOADER_CFG_BS8, DATALOADER_CFG_BS16, DATALOADER_CFG_BS32, DATALOADER_CFG_BS40
)
from .datasets import (
    DATASET_CFG_ADE20k_512x512, DATASET_CFG_ADE20k_640x640, DATASET_CFG_CITYSCAPES_512x1024, DATASET_CFG_VOCAUG_512x512,
    DATASET_CFG_ATR_473x473, DATASET_CFG_LIP_473x473, DATASET_CFG_CIHP_473x473, DATASET_CFG_PASCALCONTEXT_480x480,
    DATASET_CFG_PASCALCONTEXT59_480x480, DATASET_CFG_CHASEDB1_128x128, DATASET_CFG_DRIVE_64x64, DATASET_CFG_HRF_256x256,
    DATASET_CFG_STARE_128x128, DATASET_CFG_CITYSCAPES_1024x1024, DATASET_CFG_CITYSCAPES_832x832, DATASET_CFG_COCOStuff10k_512x512,
    DATASET_CFG_VSPW_512x512, DATASET_CFG_PASCALCONTEXT59_640x640
)