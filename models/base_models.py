import torchvision
import torch.nn as nn
import torch.nn.functional as F

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
    # TODO load pretrained weights
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

        self.conv_block_6_7 = ConvBlock(in_channels=512, out_channels=out_channels, pool=False)

        #self.load_trained_weights()


    def forward(self,x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_1(x)
        x = self.max_pool_1(x)

        x = self.conv_block_4(x)
        x = self.conv_2(x)
        conv4 = x # value will be fed to SSD ratio to input:8
        x = self.max_pool_2(x)

        x = self.conv_block_5(x)
        x = self.conv_3(x)
        x = self.max_pool_3(x)

        conv7 = self.conv_block_6_7(x)  # value will be fed to SSD ratio to input:16

        return conv4,conv7

    def load_trained_weights(self):

        state_dict = self.state_dict()
        params = list(state_dict)

        trained_state_dict = torchvision.models.vgg16_bn(pretrained=True).state_dict()
        trained_params = list(trained_state_dict)

        for idx, param in enumerate(params[0:44]):
            state_dict[param] = trained_state_dict[trained_params[idx]]

        for idx, param in enumerate(params[49:65]):
            state_dict[param] = trained_state_dict[trained_params[idx]]

        for idx, param in enumerate(params[70:86]):
            state_dict[param] = trained_state_dict[trained_params[idx]]


class SSDExtension(nn.Module):
    def __init__(self,in_channels=1024,out_channels=256):
        super(SSDExtension, self).__init__()
        self.conv_8_1 = nn.Conv2d(in_channels, 256, kernel_size=(1, 1), stride=(1, 1))
        self.conv_8_2 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        self.conv_9_1 = nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
        self.conv_9_2 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        self.conv_10_1 = nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
        self.conv_10_2 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))

        self.conv_11_1 = nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
        self.conv_11_2 = nn.Conv2d(128, out_channels, kernel_size=(3, 3), stride=(1, 1))

        self._xavier_init()

    def forward(self,x):
        x = F.relu(self.conv_8_1(x))
        x = F.relu(self.conv_8_2(x))
        conv8 = x

        x = F.relu(self.conv_9_1(x))
        x = F.relu(self.conv_9_2(x))
        conv9 = x

        x = F.relu(self.conv_10_1(x))
        x = F.relu(self.conv_10_2(x))
        conv10 = x

        x = F.relu(self.conv_11_1(x))
        x = F.relu(self.conv_11_2(x))
        conv11 = x
        return conv8,conv9,conv10,conv11

    def _xavier_init(self):
        for child in self.children():
            if isinstance(child,nn.Conv2d):
                nn.init.xavier_uniform_(child.weight)

