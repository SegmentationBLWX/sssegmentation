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
    def __init__(self, model_name, features_only=True, pretrained=True, pretrained_model_path='', in_channels=3, **kwargs):
        super(TIMMBackbone, self).__init__()
        import timm
        self.timm_model = timm.create_model(
            model_name=model_name,
            features_only=features_only,
            pretrained=pretrained,
            in_chans=in_channels,
            checkpoint_path=pretrained_model_path,
            **kwargs['extra_args'],
        )
        self.timm_model.global_pool = None
        self.timm_model.fc = None
        self.timm_model.classifier = None
    '''forward'''
    def forward(self, x):
        features = self.timm_model(x)
        return features


'''build TIMMBackbone'''
def BuildTIMMBackbone(timm_type=None, **kwargs):
    # assert whether support
    assert timm_type is None
    # parse args
    default_args = {
        'model_name': None,
        'features_only': True,
        'pretrained': True,
        'pretrained_model_path': '',
        'in_channels': 3,
        'extra_args': {},
    }
    for key, value in kwargs.items():
        if key in default_args: default_args.update({key: value})
    # obtain the instanced timm
    timm_args = default_args.copy()
    model = TIMMBackbone(**timm_args)
    # return the model
    return model