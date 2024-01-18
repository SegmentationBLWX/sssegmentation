'''
Function:
    Implementation of EMASegmentor
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn


'''EMASegmentor'''
class EMASegmentor(nn.Module):
    def __init__(self, segmentor, momentum=0.0005, device='cpu'):
        super(EMASegmentor, self).__init__()
        # set attributes
        self.device = device
        self.momentum = momentum
        # copy original segmentor
        if hasattr(segmentor, 'module'): segmentor = segmentor.module
        self.segmentor_ema = copy.deepcopy(segmentor)
        # initialize segmentor_ema
        if device: self.segmentor_ema.to(device=device)
        self.segmentor_ema.eval()
        for param in self.segmentor_ema.parameters():
            param.requires_grad = False
    '''forward'''
    def forward(self, x, targets, **kwargs):
        return self.segmentor_ema(x, targets, **kwargs)
    '''state'''
    def state(self):
        return self.segmentor_ema.state_dict()
    '''setstate'''
    def setstate(self, state_dict, strict=True):
        return self.segmentor_ema.load_state_dict(state_dict, strict=strict)
    '''update'''
    def update(self, segmentor):
        if self.device: self.segmentor_ema.to(device=self.device)
        if hasattr(segmentor, 'module'): segmentor = segmentor.module
        with torch.no_grad():
            state_dict = segmentor.state_dict()
            for ema_k, ema_v in self.segmentor_ema.state_dict().items():
                cur_v = state_dict[ema_k].detach()
                if self.device:
                    cur_v = cur_v.to(device=self.device)
                ema_v.copy_((ema_v * (1.0 - self.momentum)) + (self.momentum * cur_v))