'''
Function:
    Implementation of MaskFormer criterion
Author:
    Zhenchao Jin
'''
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


'''NestedTensor'''
class NestedTensor(object):
    def __init__(self, tensors, mask):
        self.mask = mask
        self.tensors = tensors
    '''to'''
    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask, cast_mask = self.mask, None
        if mask is not None: cast_mask = mask.to(device)
        return NestedTensor(cast_tensor, cast_mask)
    '''decompose'''
    def decompose(self):
        return self.tensors, self.mask


'''SetCriterion'''
class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        super(SetCriterion, self).__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
    '''losslabels'''
    def losslabels(self, outputs, targets, indices, num_masks):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self.getsrcpermutationidx(indices)
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        return {'loss_ce': loss_ce}
    '''lossmasks'''
    def lossmasks(self, outputs, targets, indices, num_masks):
        assert 'pred_masks' in outputs
        src_idx = self.getsrcpermutationidx(indices)
        tgt_idx = self.gettgtpermutationidx(indices)
        src_masks = outputs['pred_masks']
        src_masks = src_masks[src_idx]
        masks = [t['masks'] for t in targets]
        target_masks, valid = self.nestedtensorfromtensorlist(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]
        src_masks = F.interpolate(src_masks[:, None], size=target_masks.shape[-2:], mode='bilinear', align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)
        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            'loss_mask': self.sigmoidfocalloss(src_masks, target_masks, num_masks),
            'loss_dice': self.diceloss(src_masks, target_masks, num_masks),
        }
        return losses
    '''getsrcpermutationidx'''
    def getsrcpermutationidx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    '''gettgtpermutationidx'''
    def gettgtpermutationidx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
    '''getloss'''
    def getloss(self, loss_type, outputs, targets, indices, num_masks):
        loss_map = {'labels': self.losslabels, 'masks': self.lossmasks}
        return loss_map[loss_type](outputs, targets, indices, num_masks)
    '''formattargets'''
    def formattargets(self, seg, ignore_index=255, background_idx=0):
        labels, masks = [], []
        for label in torch.unique(seg):
            if int(label) == ignore_index: continue
            labels.append(label)
            masks.append((seg == label).unsqueeze(0))
        if not masks:
            masks = torch.zeros(1, seg.shape[0], seg.shape[1])
        else:
            masks = torch.cat(masks, dim=0).type_as(seg).long()
        if not labels: labels = [background_idx]
        labels = torch.tensor(labels).type_as(seg).long()
        return masks, labels
    '''forward'''
    def forward(self, outputs, targets):
        # format targets
        segs = targets['seg_targets']
        batch_size, targets_format = segs.shape[0], []
        for idx in range(batch_size):
            masks, labels = self.formattargets(segs[idx])
            target_format = {
                'masks': masks, 'labels': labels,
            }
            targets_format.append(target_format)
        targets = targets_format
        # forward
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)
        num_masks = sum(len(t['labels']) for t in targets)
        num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=next(iter(outputs.values())).device)
        if dist.is_available() and dist.is_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / self.getworldsize(), min=1).item()
        # losses
        losses = {}
        for loss in self.losses: losses.update(self.getloss(loss, outputs, targets, indices, num_masks))
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.getloss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses
    '''nestedtensorfromtensorlist'''
    def nestedtensorfromtensorlist(self, tensor_list):
        def maxbyaxis(the_list):
            maxes = the_list[0]
            for sublist in the_list[1:]:
                for index, item in enumerate(sublist):
                    maxes[index] = max(maxes[index], item)
            return maxes
        assert tensor_list[0].ndim == 3
        if torchvision._is_tracing(): return self.onnxnestedtensorfromtensorlist(tensor_list)
        max_size = maxbyaxis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype, device = tensor_list[0].dtype, tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
            m[:img.shape[1], :img.shape[2]] = False
        return NestedTensor(tensor, mask)
    '''onnxnestedtensorfromtensorlist'''
    @torch.jit.unused
    def onnxnestedtensorfromtensorlist(self, tensor_list):
        max_size = []
        for i in range(tensor_list[0].dim()):
            max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)).to(torch.int64)
            max_size.append(max_size_i)
        max_size, padded_imgs, padded_masks = tuple(max_size), [], []
        for img in tensor_list:
            padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
            padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
            padded_imgs.append(padded_img)
            m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
            padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), 'constant', 1)
            padded_masks.append(padded_mask.to(torch.bool))
        tensor, mask = torch.stack(padded_imgs), torch.stack(padded_masks)
        return NestedTensor(tensor, mask=mask)
    '''getworldsize'''
    def getworldsize(self):
        if (not dist.is_available()) or (not dist.is_initialized()): return 1
        return dist.get_world_size()
    '''diceloss'''
    def diceloss(self, inputs, targets, num_masks):
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1)
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / num_masks
    '''sigmoidfocalloss'''
    def sigmoidfocalloss(self, inputs, targets, num_masks, alpha=0.25, gamma=2):
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.mean(1).sum() / num_masks