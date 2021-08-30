import torch
import torch.nn as nn
import math


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(oup, eps=0.001),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )



def mobile_conv_block(inp_channel, out_channel, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d( 
                in_channels=inp_channel,
                out_channels=inp_channel,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True,
                groups=inp_channel),
            nn.BatchNorm2d(inp_channel, eps=0.001),
            nn.ReLU(inplace=True),
            nn.Conv2d( 
                in_channels=inp_channel,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True),
            nn.BatchNorm2d(out_channel, eps=0.001),
            nn.ReLU(inplace=True))


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(in_channels=inp, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim, eps=0.001),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=stride,padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim, eps=0.001),
            nn.ReLU(inplace=True),
            # pw-linear
            nn.Conv2d(in_channels=hidden_dim, out_channels=oup, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oup, eps=0.001),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, prms):
        super(MobileNetV2, self).__init__()
        width_mult = prms["alpha"]
        self.conv0 = conv_bn(3, int(width_mult*32), 2)
        self.conv1 = InvertedResidual(int(width_mult*32), int(width_mult*16), 1, expand_ratio=6)

        self.conv2_1 = InvertedResidual(int(width_mult*16), int(width_mult*24), 2, expand_ratio=6)
        self.conv2_2 = InvertedResidual(int(width_mult*24), int(width_mult*24), 1, expand_ratio=6)

        self.conv3_1 = InvertedResidual(int(width_mult*24), int(width_mult*32), 2, expand_ratio=6)
        self.conv3_2 = InvertedResidual(int(width_mult*32), int(width_mult*32), 1, expand_ratio=6)
        self.conv3_3 = InvertedResidual(int(width_mult*32), int(width_mult*32), 1, expand_ratio=6)

        self.conv4_1 = InvertedResidual(int(width_mult*32), int(width_mult*64), 2, expand_ratio=6)
        self.conv4_2 = InvertedResidual(int(width_mult*64), int(width_mult*64), 1, expand_ratio=6)
        self.conv4_3 = InvertedResidual(int(width_mult*64), int(width_mult*64), 1, expand_ratio=6)
        self.conv4_4 = InvertedResidual(int(width_mult*64), int(width_mult*64), 1, expand_ratio=6)

        self.conv5_1 = InvertedResidual(int(width_mult*64), int(width_mult*96), 1, expand_ratio=6)
        self.conv5_2 = InvertedResidual(int(width_mult*96), int(width_mult*96), 1, expand_ratio=6)
        self.conv5_3 = InvertedResidual(int(width_mult*96), int(width_mult*96), 1, expand_ratio=6)

        self.conv6_1 = InvertedResidual(int(width_mult*96), int(width_mult*160), 2, expand_ratio=6)
        self.conv6_2 = InvertedResidual(int(width_mult*160), int(width_mult*160), 1, expand_ratio=6)
        self.conv6_3 = InvertedResidual(int(width_mult*160), int(width_mult*160), 1, expand_ratio=6)

        self.conv7 = InvertedResidual(int(width_mult*160), int(width_mult*320), 1, expand_ratio=6)
        
        output_channel = 1280
        self.conv = conv_1x1_bn(int(width_mult*320), output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, 1000)
        self._initialize_weights()

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)

        x = self.conv2_1(x)
        conv_2_2 = self.conv2_2(x)

        x = self.conv3_1(conv_2_2)
        x = self.conv3_2(x)
        conv_3_3 = self.conv3_3(x)

        x = self.conv4_1(conv_3_3)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.conv4_4(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        conv_5_3 = self.conv5_3(x)

        x = self.conv6_1(conv_5_3)
        x = self.conv6_2(x)
        conv6_3 = self.conv6_3(x)

        conv7 = self.conv7(conv6_3)
        

        # classifier
        x = self.conv(conv7)
        x = self.avgpool(x)
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
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()