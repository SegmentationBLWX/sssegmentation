'''
Function:
    Implementation of Backbones Supported by timm
Author:
    Zhenchao Jin
'''
import torch.nn as nn


'''DEFAULT_MODEL_URLS'''
DEFAULT_MODEL_URLS = {}
'''AUTO_ASSERT_STRUCTURE_TYPES'''
AUTO_ASSERT_STRUCTURE_TYPES = {}


'''TIMMBackbone'''
class TIMMBackbone(nn.Module):
    def __init__(self, structure_type, model_name, features_only=True, pretrained=True, pretrained_model_path='', in_channels=3, extra_args={}):
        super(TIMMBackbone, self).__init__()
        import timm
        self.timm_model = timm.create_model(
            model_name=model_name, features_only=features_only, pretrained=pretrained, in_chans=in_channels, checkpoint_path=pretrained_model_path, **extra_args,
        )
        self.timm_model.global_pool = None
        self.timm_model.fc = None
        self.timm_model.classifier = None
    '''forward'''
    def forward(self, x):
        features = self.timm_model(x)
        return features