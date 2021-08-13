import math
import time
import os

import torch
import torch.nn as nn
from torch.nn.functional import interpolate
import numpy as np

BN_MOMENTUM = 0.1


def conv_block(inp_channel, out_channel, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d( 
                in_channels=inp_channel,
                out_channels=out_channel,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True),
            nn.BatchNorm2d(out_channel, eps=0.1),
            nn.ReLU(inplace=True))


class VGG16BN(nn.Module):
    """vgg16abstract2's copy

    Args:
        object ([type]): [description]
    """
    def __init__(self, prms):
        super(VGG16BN, self).__init__()
        alpha = prms["alpha"]
        # block1
        self.conv1 = conv_block(inp_channel=3, out_channel=int(64*alpha))
        self.conv_1_1 = conv_block(inp_channel=int(64*alpha), out_channel=int(64*alpha))
        
        # block 2
        self.conv_2_1 = conv_block(inp_channel=int(64*alpha), out_channel=int(128*alpha))
        self.conv_2_2 = conv_block(inp_channel=int(128*alpha), out_channel=int(128*alpha))

        # block 3
        self.conv_3_1 = conv_block(inp_channel=int(128*alpha), out_channel=int(256*alpha))
        self.conv_3_2 = conv_block(inp_channel=int(256*alpha), out_channel=int(256*alpha))
        self.conv_3_3 = conv_block(inp_channel=int(256*alpha), out_channel=int(256*alpha))

        # block 4
        self.conv_4_1 = conv_block(inp_channel=int(256*alpha), out_channel=int(512*alpha))
        self.conv_4_2 = conv_block(inp_channel=int(512*alpha), out_channel=int(512*alpha))
        self.conv_4_3 = conv_block(inp_channel=int(512*alpha), out_channel=int(512*alpha))

        # block 5
        self.conv_5_1 = conv_block(inp_channel=int(512*alpha), out_channel=int(512*alpha))
        self.conv_5_2 = conv_block(inp_channel=int(512*alpha), out_channel=int(512*alpha))
        self.conv_5_3 = conv_block(inp_channel=int(512*alpha), out_channel=int(512*alpha))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1000)
        )

        self._initialize_weights()


    def forward(self, x):
        # block 1
        x = self.conv1(x)
        conv_1_1 = self.conv_1_1(x)
        x = nn.MaxPool2d(kernel_size=2, stride=2)(conv_1_1)
        # block 2
        x = self.conv_2_1(x)
        conv_2_2 = self.conv_2_2(x)
        x = nn.MaxPool2d(kernel_size=2, stride=2)(conv_2_2)
        # block 3
        x = self.conv_3_1(x)
        x = self.conv_3_2(x)
        conv_3_3 = self.conv_3_3(x)
        x = nn.MaxPool2d(kernel_size=2, stride=2)(conv_3_3)
        # block 4
        x = self.conv_4_1(x)
        x = self.conv_4_2(x)
        conv_4_3 = self.conv_4_3(x)
        x = nn.MaxPool2d(kernel_size=2, stride=2)(conv_4_3)
        # block 5
        x = self.conv_5_1(x)
        x = self.conv_5_2(x)
        x = self.conv_5_3(x)
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x) # changed

        # classifier
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                # m.weight.data.zero_()
                # m.bias.data.fill_(1)

