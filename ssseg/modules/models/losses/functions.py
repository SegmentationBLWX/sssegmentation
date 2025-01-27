'''
Function:
    Implementation of calculate loss functions
Author:
    Zhenchao Jin
'''
import torch.distributed as dist
from .builder import BuildLoss


'''calculateloss'''
def calculateloss(x_src, x_tgt, loss_cfg):
    assert isinstance(loss_cfg, (dict, list, tuple))
    # calculate the loss, dict means single-type loss and list or tuple represents multiple-type losses
    if isinstance(loss_cfg, dict):
        loss = BuildLoss(loss_cfg)(x_src, x_tgt)
    else:
        loss = 0
        for l_cfg in loss_cfg:
            loss = loss + BuildLoss(l_cfg)(x_src, x_tgt)
    # return the loss
    return loss


'''calculatelosses'''
def calculatelosses(predictions, annotations, losses_cfg, preds_to_tgts_mapping=None, pixel_sampler=None):
    assert len(predictions) == len(losses_cfg), 'length of losses_cfg should be equal to the one of predictions'
    # re-organize annotations
    annotations_reorg = {}
    for pred_key in list(predictions.keys()):
        if preds_to_tgts_mapping is None:
            annotations_reorg[pred_key] = annotations['seg_targets']
        else:
            annotations_reorg[pred_key] = annotations[preds_to_tgts_mapping[pred_key]]
    annotations = annotations_reorg
    # apply pixel sampler
    if pixel_sampler is not None:
        for pred_key in list(predictions.keys()):
            predictions[pred_key], annotations[pred_key] = pixel_sampler.sample(predictions[pred_key], annotations[pred_key])
    # calculate loss according to losses_cfg
    losses_log_dict = {}
    for loss_name, loss_cfg in losses_cfg.items():
        losses_log_dict[loss_name] = calculateloss(x_src=predictions[loss_name], x_tgt=annotations[loss_name], loss_cfg=loss_cfg)
    # summarize and convert losses_log_dict
    loss_total = 0
    for loss_key, loss_value in losses_log_dict.items():
        loss_value = loss_value.mean()
        loss_total = loss_total + loss_value
        loss_value = loss_value.data.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(loss_value.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
        losses_log_dict[loss_key] = loss_value.item()
    losses_log_dict.update({'loss_total': sum(losses_log_dict.values())})
    # return the loss and losses_log_dict
    return loss_total, losses_log_dict