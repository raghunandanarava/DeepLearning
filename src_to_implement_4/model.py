import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchvision import models

class ResNet(torch.nn.Module):

    def get_bypass_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(out_channels)
        )

    def get_resblock_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=(1,1)
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=(1,1)
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def __init__(self, pool='avg'):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=(3, 3)
        )
        self.batchnorm = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3,2)
        self.block1 = self.get_resblock_layer(64, 64, 1)
        self.bypass1 = self.get_bypass_layer(64, 64, 1)

        self.block2 = self.get_resblock_layer(64, 128, 2)
        self.bypass2 = self.get_bypass_layer(64, 128, 2)

        self.block3 = self.get_resblock_layer(128, 256, 2)
        self.bypass3 = self.get_bypass_layer(128, 256, 2)

        self.block4 = self.get_resblock_layer(256, 512, 2)
        self.bypass4 = self.get_bypass_layer(256, 512, 2)

        #self.flatten = nn.Flatten()
        # self.glo_avg_pool = nn.AvgPool2d()
        self.pool = pool
        self.fc = nn.Linear(512, 2)

        if self.pool == 'ada':
            self.adaptive = nn.AdaptiveAvgPool2d((1,1))
        else:
            self.avg_pool = nn.AvgPool2d((10,10))

        self.bn = nn.BatchNorm1d(512)
        self.out = nn.Sigmoid()


    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = F.relu(x)

        x = self.maxpool(x)
        x = self.block1(x) + self.bypass1(x)
        x = self.block2(x) + self.bypass2(x)
        x = self.block3(x) + self.bypass3(x)
        x = self.block4(x) + self.bypass4(x)


        #x = F.avg_pool2d(x, x.size()[2:])
        if self.pool == 'avg':
            #x_shape = [int(s) for s in x.size()[2:]]
            #x = F.avg_pool2d(x, x_shape)
            x = self.avg_pool(x)
        else:
            x = self.adaptive(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = self.out(x)

        return x



if __name__ == '__main__':
    model = ResNet(pool='avg')
    summary(model, (3,300,300))