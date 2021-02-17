import torch
from math import sqrt


def generate_priors(image_size=300,
                    layer_sizes=None,
                    pool_ratios=None,
                    min_sizes=None,
                    max_sizes=None,
                    aspect_ratios=None):
    # TODO update feature maps, min_sizes, max_sizes for inputs size 5xx
    """
    This method generate prior boxes for SSD Model. In total, there will be 8732 prior boxes

    :param image_size: input image size for SSD Model
    :param layer_sizes: Layer sizes for each feature map
    :param pool_ratios: pooling ratio for each feature map.
        layer_size*pool_ratio = image_size
    :param min_sizes: minimum prior box size
    :param max_sizes: maximum prior box size
    :param aspect_ratios: ratio for prior box height and width

    :return: tensor of prior boxes
    """
    if aspect_ratios is None:
        aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    if min_sizes is None:
        min_sizes = [30, 60, 111, 162, 213, 264]
    if max_sizes is None:
        max_sizes = [60, 111, 162, 213, 264, 315]
    if pool_ratios is None:
        pool_ratios = [8, 16, 32, 64, 100, 300]
    if layer_sizes is None:
        layer_sizes = [38, 19, 10, 5, 3, 1]

    boxes = []
    for layer_size_idx, layer_size in enumerate(layer_sizes):
        min_size = min_sizes[layer_size_idx]
        max_size = max_sizes[layer_size_idx]
        pool_ratio = pool_ratios[layer_size_idx]
        for layer_height in range(layer_size):
            for layer_width in range(layer_size):

                layer_image_size = image_size / pool_ratio

                center_dim_x = (layer_width + 0.5) / layer_image_size
                center_dim_y = (layer_height + 0.5) / layer_image_size

                layer_min_size = min_size / image_size
                boxes += [center_dim_x, center_dim_y, layer_min_size, layer_min_size]

                diagonal = sqrt(layer_min_size * (max_size/image_size))
                boxes += [center_dim_x, center_dim_y, diagonal, diagonal]

                for ar in aspect_ratios[layer_size_idx]:
                    boxes += [center_dim_x, center_dim_y, layer_min_size * sqrt(ar), layer_min_size / sqrt(ar)]
                    boxes += [center_dim_x, center_dim_y, layer_min_size / sqrt(ar), layer_min_size * sqrt(ar)]

    output = torch.Tensor(boxes).view(-1, 4).clamp_(min=0, max=1)
    output.clamp_(max=1, min=0)
    return output

# TODO create an encoder-decoder class to wrap following methods
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

def decode(result,nms_thres):
    pass
