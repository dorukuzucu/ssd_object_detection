from torch import nn
import numpy as np
import torch


def cxcy_to_xy(box):
    """
    :param box: input boxes. Should be in center sized coordinates and size of (N,4) where N=number of priors
    :return: center coordinate boxes converted to (x_min,y_min,x_max,y_max) coordinates
    """
    start = box[:, :2] - (box[:, 2:] / 2)
    end = box[:, :2] + (box[:, 2:] / 2)

    return torch.cat([start,end],dim=1)

def xy_to_cxcy(box):
    """
    :param box: input boxes. Should be in (x_min,y_min,x_max,y_max) and size of (N,4) where N=number of priors.
    :return: input boxes converted to center
    """
    center = (box[:, 2:] + box[:, :2]) / 2
    h_w = (box[:, 2:] - box[:, :2])

    return torch.cat([center, h_w], dim=1)

def encode(cxcy,priors):
    g_cxcy = (cxcy[:, :2] - priors[:, :2]) / (priors[:, 2:] / 10)
    g_wh = torch.log(cxcy[:, 2:] / priors[:, 2:]) * 5
    return torch.cat([g_cxcy,g_wh], 1)


class IOUMetric:
    def __init__(self):
        pass

    def calculate_intersection(self,preds,target):
        min_indexes = torch.max(preds[:, :2].unsqueeze(1), target[:, :2].unsqueeze(0))
        max_indexes = torch.min(preds[:, 2:].unsqueeze(1), target[:, 2:].unsqueeze(0))

        diff = torch.clamp(max_indexes-min_indexes,min=0)

        return diff[:,:,0] * diff[:,:,1]

    def calculate_union(self,preds,target):
        intersection = self.calculate_intersection(preds=preds,target=target)

        pred_areas = (preds[:, 2] - preds[:, 0]) * (preds[:, 3] - preds[:, 1])
        target_areas = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])

        return pred_areas.unsqueeze(1) + target_areas.unsqueeze(0) - intersection


    def calculate_iou(self,preds,target):

        intersection = self.calculate_intersection(preds,target)
        union = self.calculate_union(preds,target)

        return intersection / union


class MultiboxLoss():
    def __init__(self,num_cls,overlap,priors,gpu_flag=True):
        super(LocalizationLoss, self).__init__()
        self.num_cls = num_cls
        self.overlap = overlap
        self.priors = priors
        self.gpu_flag = gpu_flag
        self.iou = IOUMetric()
        self.l1 = nn.L1Loss

    def calculate(self,preds,labels):
        """
        :param preds: predictions. location: (batch_size, num_priors, 4), scores: (batch_size, num_priors, num_cls)
        :param labels: dictionary contains 2 keys: 'labels','boxes'. labels: (batch_size,N,1), boxes: (batch_size,N,4)
        :return: Multibox loss value as float tensor

        PS: preds and labels should have percentage indexes.
        """
        # TODO convert priors cx,cy to x,y coordinates

        location, scores = preds
        classes = labels["labels"]
        b_box = labels["boxes"]

        assert location.size(0)==scores.size(0)==labels.size(0), "Mismatch on batch sizes"
        assert location.size(2)==b_box.size(2), "Missing bounding box info"
        assert scores.size(2)==self.num_cls, "Number of classes do not match"

        num_priors = scores.size(2)
        batch_size = scores.size(0)

        true_locs = torch.zeros((batch_size, num_priors, 4), dtype=torch.float)
        true_classes = torch.zeros((batch_size, num_priors), dtype=torch.long)

        for batch_idx in range(batch_size):
            overlap_mat = self.iou.calculate_iou(b_box[batch_idx],self.priors)
            iou_score, cls = overlap_mat.max(dim=0)
            best_score, best_prior_idx = overlap_mat.max(dim=1)

            for idx in range(best_prior_idx.size(0)):
                cls[best_prior_idx[idx]] = idx

            labels_for_priors = classes[batch_idx][cls]
            labels_for_priors[best_prior_idx<self.overlap] = 0

            true_classes[batch_idx] = labels_for_priors
            true_locs[batch_idx] = encode(labels[batch_idx],self.priors)

        positive_priors = (true_classes!=0)

        return self.l1(location[positive_priors],true_locs[positive_priors])

class ConfidenceLoss():
    def __init__(self):
        pass
