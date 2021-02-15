import torch
import torch.nn as nn
from models.utils import generate_priors
from models.base_models import VGG16,SSDExtension


class DetectionLayer(nn.Module):
    # TODO parameterize input and output channels
    def __init__(self,class_no):
        super(DetectionLayer, self).__init__()
        self.class_no = class_no

        self.conv4 = nn.Conv2d(512, 4*self.class_no, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv7 = nn.Conv2d(1024, 6*self.class_no, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv8 = nn.Conv2d(512, 6*self.class_no, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv9 = nn.Conv2d(256, 6*self.class_no, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv10 = nn.Conv2d(256, 4*self.class_no, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv11 = nn.Conv2d(256, 4*self.class_no, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self._xavier_init()

    def forward(self,conv4,conv7,conv8,conv9,conv10,conv11):
        outs = []
        out_conv4 = self.conv4(conv4)
        out_conv4 = out_conv4.permute(0, 2, 3, 1).contiguous()
        out_conv4 = out_conv4.view(out_conv4.size(0), -1, self.class_no)
        outs.append(out_conv4)

        out_conv7 = self.conv7(conv7)
        out_conv7 = out_conv7.permute(0, 2, 3, 1).contiguous()
        out_conv7 = out_conv7.view(out_conv7.size(0), -1, self.class_no)
        outs.append(out_conv7)

        out_conv8 = self.conv8(conv8)
        out_conv8 = out_conv8.permute(0, 2, 3, 1).contiguous()
        out_conv8 = out_conv8.view(out_conv8.size(0), -1, self.class_no)
        outs.append(out_conv8)

        out_conv9 = self.conv9(conv9)
        out_conv9 = out_conv9.permute(0, 2, 3, 1).contiguous()
        out_conv9 = out_conv9.view(out_conv9.size(0), -1, self.class_no)
        outs.append(out_conv9)

        out_conv10 = self.conv10(conv10)
        out_conv10 = out_conv10.permute(0, 2, 3, 1).contiguous()
        out_conv10 = out_conv10.view(out_conv10.size(0), -1, self.class_no)
        outs.append(out_conv10)

        out_conv11 = self.conv11(conv11)
        out_conv11 = out_conv11.permute(0, 2, 3, 1).contiguous()
        out_conv11 = out_conv11.view(out_conv11.size(0), -1, self.class_no)
        outs.append(out_conv11)
        return torch.cat(outs,1).contiguous()

    def _xavier_init(self):
        for child in self.children():
            if isinstance(child,nn.Conv2d):
                nn.init.xavier_uniform_(child.weight)


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
        self.priors = generate_priors(300)

        self.vgg_backbone = VGG16(in_channels=3,out_channels=1024)
        self.ssd_extension = SSDExtension()
        self.loc_layers = DetectionLayer(4) # produces bounding box coordinates
        self.conf_layers = DetectionLayer(self.num_classes) # Produces confidence scores for each class

    def forward(self,x):
        conv4, conv7 = self.vgg_backbone(x)
        factor = conv4.pow(2).sum(dim=1, keepdim=True).sqrt()
        rescale_factor = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))
        conv4 = conv4 * rescale_factor / factor

        conv8,conv9,conv10,conv11 = self.ssd_extension(conv7)

        locations = self.loc_layers(conv4,conv7,conv8,conv9,conv10,conv11)
        scores = self.conf_layers(conv4, conv7, conv8, conv9, conv10, conv11)

        return locations,scores
