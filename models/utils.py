from itertools import product
import torch
from math import sqrt


def generate_priors(image_size=300):
  pass


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