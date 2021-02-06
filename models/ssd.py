import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os

from models.utils import generate_anchors
from models.base_models import VGG16,SSDExtension

def loc_layers():
    return nn.Sequential(
        nn.Conv2d(512, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.Conv2d(1024, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.Conv2d(512, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.Conv2d(256, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.Conv2d(256, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.Conv2d(256, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )

def conf_layers():
    return nn.Sequential(
        nn.Conv2d(512, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.Conv2d(1024, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.Conv2d(512, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.Conv2d(256, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.Conv2d(256, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.Conv2d(256, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )


class SSD(nn.Module):
    def __init__(self,
                 state,
                 ssd_model,
                 num_classes
                 ):
        super(SSD, self).__init__()

        self.state = state
        self.num_classes = num_classes
        self.size = ssd_model

        self.anchors = generate_anchors()
        self.loc_layers = loc_layers()
        self.conf_layers = conf_layers()

        self.vgg_backbone = VGG16(in_channels=3,out_channels=1024)
        self.ssd_extension = SSDExtension(in_channels=1024,out_channels=256)

    def forward(self):
        pass