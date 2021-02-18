from torch import nn
from models.utils import *


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
    def __init__(self,num_cls,priors=None,overlap=0.5,hard_neg_ratio=3,alpha=1,gpu_flag=True):
        super(MultiboxLoss, self).__init__()
        self.num_cls = num_cls
        self.priors = priors if priors is not None else generate_priors()
        self.overlap = overlap
        self.hard_neg_ratio = hard_neg_ratio
        self.alpha = alpha
        self.gpu_flag = gpu_flag
        self.iou = IOUMetric()
        self.l1 = nn.L1Loss()
        self.ce = nn.CrossEntropyLoss(reduce=False)

    def calculate(self, preds, target):
        """
        :param preds: predictions. location: (batch_size, num_priors, 4), scores: (batch_size, num_priors, num_cls)
        :param target: dictionary contains 2 keys: 'labels','boxes'. labels: (batch_size,N,1), boxes: (batch_size,N,4)
        :return: Multibox loss value as float tensor

        PS: preds and labels should have percentage indexes.
        """
        # TODO convert priors cx,cy to x,y coordinates
        ########################################################################
        ###################### Localization Loss ###############################
        ########################################################################

        location, scores = preds
        classes = target["labels"]
        b_box = target["boxes"]

        assert location.size(0)==scores.size(0)==len(classes), "Mismatch on batch sizes"
        assert scores.size(2)==self.num_cls, "Number of classes do not match"

        num_priors = self.priors.size(0)
        batch_size = scores.size(0)

        true_locs = torch.zeros((batch_size, num_priors, 4), dtype=torch.float)
        true_classes = torch.zeros((batch_size, num_priors), dtype=torch.long)

        for batch_idx in range(batch_size):
            overlap_mat = self.iou.calculate_iou(b_box[batch_idx],self.priors)
            iou_score_prior, cls_prior = overlap_mat.max(dim=0) # 8732
            iou_score_target, prior_for_target = overlap_mat.max(dim=1) # number of object

            for idx in range(prior_for_target.size(0)):
                cls_prior[prior_for_target[idx]] = idx

            labels_for_priors = classes[batch_idx][cls_prior]

            for idx in range(prior_for_target.size(0)):
                if prior_for_target[idx] < self.overlap:
                    labels_for_priors[prior_for_target[idx]] = 0
                iou_score_prior[prior_for_target[idx]] = 1

            true_classes[batch_idx] = labels_for_priors
            true_locs[batch_idx] = encode(b_box[batch_idx][cls_prior], self.priors)

        positive_priors = (true_classes!=0)

        localization_loss = self.l1(location[positive_priors],true_locs[positive_priors])

        ########################################################################
        ######################## Confidence Loss ###############################
        ########################################################################

        num_positives = positive_priors.sum(dim=1) #number of positives
        num_negatives = self.hard_neg_ratio * num_positives # number of hard negatives

        conf_loss = self.ce(scores.view(-1,self.num_cls),true_classes.view(-1))
        conf_loss = conf_loss.view(batch_size,num_priors)
        pos_conf_loss = conf_loss[positive_priors]

        neg_conf_loss = conf_loss.clone()  # (N, 8732)
        neg_conf_loss[positive_priors] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
        neg_conf_loss, _ = neg_conf_loss.sort(dim=1, descending=True)  # (N, 8732), sorted by decreasing hardness
        hardness_ranks = torch.LongTensor(range(num_priors)).unsqueeze(0).expand_as(neg_conf_loss)
        hard_negatives = hardness_ranks < num_negatives.unsqueeze(1)
        conf_loss_hard_neg = neg_conf_loss[hard_negatives]

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = (conf_loss_hard_neg.sum() + pos_conf_loss.sum()) / num_positives.sum().float()  # (), scalar

        # TOTAL LOSS
        return conf_loss + self.alpha * localization_loss
