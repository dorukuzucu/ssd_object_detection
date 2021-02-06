import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init


def conv2d_block(in_channels,out_channels, bn=False):
    """
    :param in_channels: number of input channels for conv net, int
    :param out_channels: number of output channels for conv net, int
    :param bn: batch normalization flag, boolean. Adds a batch norm layer between conv and Relu if bn is set to True
    :return: Sequential layers, sub-network consists of conv bn relu
    """
    layers = []
    layers.append(nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(3,3),stride=1,padding=1))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class ConvBlock(nn.Module):
    """
        Convolutional Block
    """
    def __init__(self, in_channels,out_channels, bn=True, pool=True):
        super().__init__()
        self.pool = pool
        # first block on conv-bn-relu
        self.conv_block_1 = conv2d_block(in_channels=in_channels,out_channels=out_channels,bn=bn)
        self.conv_block_2 = conv2d_block(in_channels=out_channels,out_channels=out_channels,bn=bn)

        if self.pool:
            self.max_pool = nn.MaxPool2d(kernel_size=2,stride=2)

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)

        if self.pool:
            x = self.max_pool(x)

        return x


class VGG16(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(VGG16, self).__init__()
        self.conv_block_1 = ConvBlock(in_channels=in_channels,out_channels=64)
        self.conv_block_2 = ConvBlock(in_channels=64,out_channels=128)
        self.conv_block_3 = ConvBlock(in_channels=128,out_channels=256,pool=False)
        self.conv_1 = conv2d_block(in_channels=256,out_channels=256)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv_block_4 = ConvBlock(in_channels=256,out_channels=512,pool=False)
        self.conv_2 = conv2d_block(in_channels=512,out_channels=512)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_block_5 = ConvBlock(in_channels=512,out_channels=512,pool=False)
        self.conv_3 = conv2d_block(in_channels=512,out_channels=512)
        self.max_pool_3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.conv_block_6 = ConvBlock(in_channels=512, out_channels=out_channels, pool=False)


    def forward(self,x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_1(x)
        x = self.max_pool_1(x)

        x = self.conv_block_4(x)
        x = self.conv_2(x)
        conv4_3 = x # get value which will be fed to SSD
        x = self.max_pool_2(x)

        x = self.conv_block_5(x)
        x = self.conv_3(x)
        x = self.max_pool_3(x)

        x = self.conv_block_6(x)

        return x,conv4_3


class SSDExtension(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(SSDExtension, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, 256, kernel_size=(1, 1), stride=(1, 1))
        self.conv_2 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv_3 = nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
        self.conv_4 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv_5 = nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
        self.conv_6 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))
        self.conv_7 = nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
        self.conv_8 = nn.Conv2d(128, out_channels, kernel_size=(3, 3), stride=(1, 1))

    def forward(self,x):
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        conv2 = x
        x = F.relu(self.conv_3(x))
        x = F.relu(self.conv_4(x))
        conv4 = x
        x = F.relu(self.conv_5(x))
        x = F.relu(self.conv_6(x))
        conv6 = x
        x = F.relu(self.conv_7(x))
        x = F.relu(self.conv_8(x))

        return x,(conv2,conv4,conv6)


"""
vgg = VGG16(3,1024)
print(vgg)

ssd = SSDExtension(1024,256)
print(ssd)

x = torch.rand((1,3,512,512))
out1 = vgg(x)

print(out1.size())
out2 = ssd(out1)
print(out2.size())
"""

