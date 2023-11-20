'''
Function:
    Implementation of Expectation Maximization Attention Module
Author:
    Zhenchao Jin
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


'''EMAModule'''
class EMAModule(nn.Module):
    def __init__(self, channels, num_bases, num_stages, momentum):
        super(EMAModule, self).__init__()
        assert num_stages >= 1, 'num_stages must be at least 1'
        self.num_bases = num_bases
        self.num_stages = num_stages
        self.momentum = momentum
        bases = torch.zeros(1, channels, self.num_bases)
        bases.normal_(0, math.sqrt(2. / self.num_bases))
        bases = F.normalize(bases, dim=1, p=2)
        self.register_buffer('bases', bases)
    '''forward'''
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        feats = x.view(batch_size, channels, height * width)
        bases = self.bases.repeat(batch_size, 1, 1)
        with torch.no_grad():
            for _ in range(self.num_stages):
                attention = torch.einsum('bcn,bck->bnk', feats, bases)
                attention = F.softmax(attention, dim=2)
                attention_normed = F.normalize(attention, dim=1, p=1)
                bases = torch.einsum('bcn,bnk->bck', feats, attention_normed)
                bases = F.normalize(bases, dim=1, p=2)
        feats_recon = torch.einsum('bck,bnk->bcn', bases, attention)
        feats_recon = feats_recon.view(batch_size, channels, height, width)
        if self.training:
            bases = bases.mean(dim=0, keepdim=True)
            bases = self.reducemean(bases)
            bases = F.normalize(bases, dim=1, p=2)
            self.bases = (1 - self.momentum) * self.bases + self.momentum * bases
        return feats_recon
    '''reduce mean when distributed training'''
    def reducemean(self, tensor):
        if not (dist.is_available() and dist.is_initialized()):
            return tensor
        tensor = tensor.clone()
        dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
        return tensor