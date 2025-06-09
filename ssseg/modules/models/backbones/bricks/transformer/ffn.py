'''
Function:
    Implementation of FFN
Author:
    Zhenchao Jin
'''
import torch.nn as nn
from ..misc import LayerScale
from ..dropout import BuildDropout
from ..activation import BuildActivation


'''FFN'''
class FFN(nn.Module):
    def __init__(self, embed_dims=256, feedforward_channels=1024, num_fcs=2, act_cfg=dict(type='ReLU', inplace=True), ffn_drop=0., dropout_cfg=None, add_identity=True, layer_scale_init_value=0.):
        super(FFN, self).__init__()
        assert num_fcs >= 2, f'num_fcs should be no less than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(nn.Sequential(
                nn.Linear(in_channels, feedforward_channels), BuildActivation(act_cfg=act_cfg), nn.Dropout(ffn_drop)
            ))
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)
        self.dropout_layer = BuildDropout(dropout_cfg) if dropout_cfg else nn.Identity()
        self.add_identity = add_identity
        if layer_scale_init_value > 0:
            self.gamma2 = LayerScale(embed_dims, scale=layer_scale_init_value)
        else:
            self.gamma2 = nn.Identity()
    '''forward'''
    def forward(self, x, identity=None):
        out = self.layers(x)
        out = self.gamma2(out)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)