'''
Function:
    Implementation of Evaluation
Author:
    Zhenchao Jin
'''
import torch
import numpy as np
from collections import OrderedDict


'''Evaluation'''
class Evaluation():
    def __init__(self, seg_preds, seg_targets, num_classes, ignore_index=-1, nan_to_num=None, beta=1.0):
        total_area_intersect, total_area_union, total_area_pred_label, total_area_label = self.totalintersectandunion(
            results=seg_preds,
            gt_seg_maps=seg_targets,
            num_classes=num_classes,
            ignore_index=ignore_index
        )
        self.all_metric_results = self.totalareatometrics(
            total_area_intersect=total_area_intersect, 
            total_area_union=total_area_union, 
            total_area_pred_label=total_area_pred_label, 
            total_area_label=total_area_label, 
            nan_to_num=nan_to_num, 
            beta=beta,
        )
    '''calculate total intersection and union'''
    @staticmethod
    def totalintersectandunion(results, gt_seg_maps, num_classes, ignore_index=-1):
        total_area_intersect = torch.zeros((num_classes, ), dtype=torch.float64)
        total_area_union = torch.zeros((num_classes, ), dtype=torch.float64)
        total_area_pred_label = torch.zeros((num_classes, ), dtype=torch.float64)
        total_area_label = torch.zeros((num_classes, ), dtype=torch.float64)
        for result, gt_seg_map in zip(results, gt_seg_maps):
            area_intersect, area_union, area_pred_label, area_label = Evaluation.intersectandunion(result, gt_seg_map, num_classes, ignore_index)
            total_area_intersect += area_intersect
            total_area_union += area_union
            total_area_pred_label += area_pred_label
            total_area_label += area_label
        return total_area_intersect, total_area_union, total_area_pred_label, total_area_label
    '''calculate intersection and union'''
    @staticmethod
    def intersectandunion(pred_label, label, num_classes, ignore_index=-1):
        # convert to torch.array
        pred_label = torch.from_numpy(pred_label)
        label = torch.from_numpy(label)
        # filter
        mask = (label != ignore_index)
        pred_label = pred_label[mask]
        label = label[mask]
        # calculate
        intersect = pred_label[pred_label == label]
        area_intersect = torch.histc(intersect.float(), bins=(num_classes), min=0, max=num_classes-1)
        area_pred_label = torch.histc(pred_label.float(), bins=(num_classes), min=0, max=num_classes-1)
        area_label = torch.histc(label.float(), bins=(num_classes), min=0, max=num_classes-1)
        area_union = area_pred_label + area_label - area_intersect
        # return
        return area_intersect, area_union, area_pred_label, area_label
    '''calculate evaluation metrics'''
    @staticmethod
    def totalareatometrics(total_area_intersect, total_area_union, total_area_pred_label, total_area_label, nan_to_num=None, beta=1):
        # all metrics
        recall = total_area_intersect / total_area_label
        precision = total_area_intersect / total_area_pred_label
        fscore = torch.tensor([Evaluation.calcuatefscore(x[0], x[1], beta) for x in zip(precision, recall)])
        all_metric_results = OrderedDict({
            'fscore': fscore,
            'recall': recall,
            'precision': precision,
            'iou': total_area_intersect / total_area_union,
            'accuracy': total_area_intersect / total_area_label,
            'all_accuracy': total_area_intersect.sum() / total_area_label.sum(),
            'dice': 2 * total_area_intersect / (total_area_pred_label + total_area_label),
        })
        # convert to numpy
        all_metric_results = {
            metric: metric_value.numpy() for metric, metric_value in all_metric_results.items()
        }
        # nan to num
        if nan_to_num is not None:
            all_metric_results = OrderedDict({
                metric: np.nan_to_num(metric_value, nan=nan_to_num) for metric, metric_value in all_metric_results.items()
            })
        # return
        all_metric_results['miou'] = all_metric_results['iou'].mean()
        all_metric_results['mdice'] = all_metric_results['dice'].mean()
        all_metric_results['mfscore'] = all_metric_results['fscore'].mean()
        return all_metric_results
    '''calcuate the f-score value'''
    @staticmethod
    def calcuatefscore(precision, recall, beta=1):
        score = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
        return score