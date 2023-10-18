'''
Function:
    Implementation of some utils
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


'''calculateuncertainty'''
def calculateuncertainty(logits):
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


'''getclones'''
def getclones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


'''pointsample'''
def pointsample(inputs, point_coords, **kwargs):
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(inputs, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


'''getuncertainpointcoordswithrandomness'''
def getuncertainpointcoordswithrandomness(coarse_logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio):
    # assert
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    # get uncertain point coords with randomness
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = pointsample(coarse_logits, point_coords, align_corners=False)
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(num_boxes, num_uncertain_points, 2)
    if num_random_points > 0:
        point_coords = torch.cat([point_coords, torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device)], dim=1)
    # return outputs
    return point_coords