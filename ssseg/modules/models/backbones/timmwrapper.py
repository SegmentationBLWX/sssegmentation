'''
Function:
    Implementation of Backbones Supported by timm
Author:
    Zhenchao Jin
'''
import torch.nn as nn


'''model urls'''
model_urls = {}


'''TIMMBackbone'''
class TIMMBackbone(nn.Module):
    def __init__(self, model_name, features_only=True, pretrained=True, pretrained_model_path='', in_channels=3, extra_args={}):
        super(TIMMBackbone, self).__init__()
        import timm
        self.timm_model = timm.create_model(
            model_name=model_name,
            features_only=features_only,
            pretrained=pretrained,
            in_chans=in_channels,
            checkpoint_path=pretrained_model_path,
            **extra_args,
        )
        self.timm_model.global_pool = None
        self.timm_model.fc = None
        self.timm_model.classifier = None
    '''forward'''
    def forward(self, x):
        features = self.timm_model(x)
        return features


'''BuildTIMMBackbone'''
def BuildTIMMBackbone(timm_cfg):
    # assert whether support
    timm_type = timm_cfg.pop('type')
    # parse cfg
    default_cfg = {
        'model_name': None,
        'features_only': True,
        'pretrained': True,
        'pretrained_model_path': '',
        'in_channels': 3,
        'extra_args': {},
    }
    for key, value in timm_cfg.items():
        if key in default_cfg: 
            default_cfg.update({key: value})
    # obtain timm_cfg
    timm_cfg = default_cfg.copy()
    # obtain the instanced timm
    model = TIMMBackbone(**timm_cfg)
    # return the model
    return model