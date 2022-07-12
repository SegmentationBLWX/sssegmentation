'''
Function:
    Implementation of ANNNet
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
from ..base import BaseModel
from .afnblock import AFNBlock
from .apnblock import APNBlock
from ...backbones import BuildActivation, BuildNormalization, constructnormcfg


'''ANNNet'''
class ANNNet(BaseModel):
    def __init__(self, cfg, mode):
        super(ANNNet, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg = self.align_corners, self.norm_cfg, self.act_cfg
        # build AFNBlock
        afnblock_cfg = cfg['afnblock']
        self.afn_block = AFNBlock(
            low_in_channels=afnblock_cfg['low_in_channels'],
            high_in_channels=afnblock_cfg['high_in_channels'], 
            transform_channels=afnblock_cfg['transform_channels'], 
            out_channels=afnblock_cfg['out_channels'], 
            query_scales=afnblock_cfg['query_scales'], 
            key_pool_scales=afnblock_cfg['key_pool_scales'],
            norm_cfg=copy.deepcopy(norm_cfg),
            act_cfg=copy.deepcopy(act_cfg),
        )
        # build APNBlock
        apnblock_cfg = cfg['apnblock']
        self.apn_block = APNBlock(
            in_channels=apnblock_cfg['in_channels'], 
            transform_channels=apnblock_cfg['transform_channels'], 
            out_channels=apnblock_cfg['out_channels'], 
            query_scales=apnblock_cfg['query_scales'], 
            key_pool_scales=apnblock_cfg['key_pool_scales'],
            norm_cfg=copy.deepcopy(norm_cfg),
            act_cfg=copy.deepcopy(act_cfg),
        )
        # build bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(afnblock_cfg['out_channels'], apnblock_cfg['in_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(constructnormcfg(placeholder=apnblock_cfg['in_channels'], norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
        )
        # build decoder
        decoder_cfg = cfg['decoder']
        self.decoder = nn.Sequential(
            nn.Dropout2d(decoder_cfg['dropout']),
            nn.Conv2d(decoder_cfg['in_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        # build auxiliary decoder
        self.setauxiliarydecoder(cfg['auxiliary'])
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
    '''forward'''
    def forward(self, x, targets=None, losses_cfg=None):
        img_size = x.size(2), x.size(3)
        # feed to backbone network
        backbone_outputs = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))
        low_feats, high_feats = backbone_outputs[-2], backbone_outputs[-1]
        # feed to AFNBlock
        feats = self.afn_block(low_feats, high_feats)
        feats = self.decoder[0](feats)
        # feed to bottleneck
        feats = self.bottleneck(feats)
        # feed to APNBlock
        feats = self.apn_block(feats)
        # feed to decoder
        predictions = self.decoder[1](feats)
        # forward according to the mode
        if self.mode == 'TRAIN':
            loss, losses_log_dict = self.forwardtrain(
                predictions=predictions,
                targets=targets,
                backbone_outputs=backbone_outputs,
                losses_cfg=losses_cfg,
                img_size=img_size,
            )
            return loss, losses_log_dict
        return predictions
    '''return all layers'''
    def alllayers(self):
        all_layers = {
            'backbone_net': self.backbone_net,
            'afn_block': self.afn_block,
            'apn_block': self.apn_block,
            'bottleneck': self.bottleneck,
            'decoder': self.decoder,
        }
        if hasattr(self, 'auxiliary_decoder'):
            all_layers['auxiliary_decoder'] = self.auxiliary_decoder
        return all_layers