'''
Function:
    Implementation of FFN
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
from ..dropout import BuildDropout
from ..activation import BuildActivation


'''FFN'''
class FFN(nn.Module):
    def __init__(self, embed_dims=256, feedforward_channels=1024, num_fcs=2, act_cfg=None, ffn_drop=0., dropout_cfg=None, add_identity=True):
        super(FFN, self).__init__()
        assert num_fcs >= 2, f'num_fcs should be no less than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = BuildActivation(act_cfg)
        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(nn.Sequential(
                nn.Linear(in_channels, feedforward_channels),
                self.activate,
                nn.Dropout(ffn_drop)
            ))
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)
        if dropout_cfg:
            self.dropout_layer = BuildDropout(dropout_cfg)
        else:
            self.dropout_layer = torch.nn.Identity()
        self.add_identity = add_identity
    '''forward'''
    def forward(self, x, identity=None):
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)