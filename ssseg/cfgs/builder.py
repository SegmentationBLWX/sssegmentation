'''
Function:
    build config file
Author:
    Zhenchao Jin
'''
import importlib


'''build config'''
def BuildConfig(modelname, datasetname, backbonename, **kwargs):
    modelname, datasetname, backbonename = modelname.lower(), datasetname.lower(), backbonename.lower()
    supported_dict = {
                'fcn': [['voc', 'resnet50os8'], ['voc', 'resnet101os8'], ['voc', 'resnet50os16'], ['voc', 'resnet101os16'],
                        ['ade20k', 'resnet50os8'], ['ade20k', 'resnet101os8'], ['ade20k', 'resnet50os16'], ['ade20k', 'resnet101os16'], 
                        ['cityscapes', 'resnet50os8'], ['cityscapes', 'resnet101os8'], ['cityscapes', 'resnet50os16'], ['cityscapes', 'resnet101os16']],
                'ce2p': [['voc', 'resnet50os8'], ['voc', 'resnet101os8'], ['voc', 'resnet50os16'], ['voc', 'resnet101os16'],
                         ['lip', 'resnet50os8'], ['lip', 'resnet101os8'], ['lip', 'resnet50os16'], ['lip', 'resnet101os16'],
                         ['atr', 'resnet50os8'], ['atr', 'resnet101os8'], ['atr', 'resnet50os16'], ['atr', 'resnet101os16'],
                         ['cihp', 'resnet50os8'], ['cihp', 'resnet101os8'], ['cihp', 'resnet50os16'], ['cihp', 'resnet101os16']],
                'ocrnet': [['voc', 'resnet50os8'], ['voc', 'resnet101os8'], ['voc', 'resnet50os16'], ['voc', 'resnet101os16'],
                           ['lip', 'resnet50os8'], ['lip', 'resnet101os8'], ['lip', 'resnet50os16'], ['lip', 'resnet101os16'],
                           ['atr', 'resnet50os8'], ['atr', 'resnet101os8'], ['atr', 'resnet50os16'], ['atr', 'resnet101os16'],
                           ['cihp', 'resnet50os8'], ['cihp', 'resnet101os8'], ['cihp', 'resnet50os16'], ['cihp', 'resnet101os16']],
                'pspnet': [['voc', 'resnet50os8'], ['voc', 'resnet101os8'], ['voc', 'resnet50os16'], ['voc', 'resnet101os16'],
                           ['ade20k', 'resnet50os8'], ['ade20k', 'resnet101os8'], ['ade20k', 'resnet50os16'], ['ade20k', 'resnet101os16'], 
                           ['cityscapes', 'resnet50os8'], ['cityscapes', 'resnet101os8'], ['cityscapes', 'resnet50os16'], ['cityscapes', 'resnet101os16']],
                'deeplabv3plus': [['voc', 'resnet50os8'], ['voc', 'resnet101os8'], ['voc', 'resnet50os16'], ['voc', 'resnet101os16'],
                                  ['ade20k', 'resnet50os8'], ['ade20k', 'resnet101os8'], ['ade20k', 'resnet50os16'], ['ade20k', 'resnet101os16'],
                                  ['cityscapes', 'resnet50os8'], ['cityscapes', 'resnet101os8'], ['cityscapes', 'resnet50os16'], ['cityscapes', 'resnet101os16']],
            }
    assert modelname in supported_dict, 'unsupport modelname %s...' % modelname
    model_supported_list = supported_dict[modelname]
    assert [datasetname, backbonename] in model_supported_list, 'unsupport datasetname %s and backbonename %s when using %s...' % (datasetname, backbonename, modelname)
    cfg = importlib.import_module(f'.{modelname}.cfgs_{datasetname}_{backbonename}', __package__)
    assert (modelname == cfg.MODEL_CFG['type']) and (datasetname == cfg.DATASET_CFG['train']['type']) and (cfg.MODEL_CFG['backbone']['type'] in backbonename), 'it seems parse cfg error...'
    return cfg, f'cfgs/{modelname}/cfgs_{datasetname}_{backbonename}.py'