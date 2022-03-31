# Pytorch官方实现的resnet： https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

import torch
import torch.nn as nn


# resnet18 and resnet34 block
class block_18_34(nn.Module):
    expansion = 1

    # 为了兼容 block_50_101_152 中的expansion，方便后边动态调用，定义一个expansion，

    # 18层，34层的残差结构中，也会有虚线残差结构，也就是进行下采样，调整尺寸以及通道数。
    # 用stride来控制是否下采样，stride=1：不用下采样，对应的就是实线残差结构；
    #                       stride=2：需要下采样，对应的就是虚线残差结构。同时也需要传入downsample。

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(block_18_34, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += identity
        x = self.relu(x)
        return x


# resnet50, resnet101 and resnet34 residual
class block_50_101_152(nn.Module):
    # 因为50层，101层，152层的残差结构中，block中卷积的通道数是不一样的。
    # 都是，最后一个1*1的卷积的通道数，是前两个卷积通道数的四倍。
    # 所以需要用一个expansion参数，来指定最后一个1*1卷积的通道数。
    expansion = 4

    # 用stride来控制是否下采样，stride=1：不用下采样，对应的就是实线残差结构；
    #                       stride=2：需要下采样，对应的就是虚线残差结构。同时也需要传入downsample。
    def __init__(self, in_channel, out_channel, stride=1,
                 downsample=None):
        super(block_50_101_152, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, blocks_num, num_classes=1000, include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        # conv1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channel,
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)

        # conv2_x
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])

        # conv3_x
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)

        # conv4_x
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)

        # conv5_x
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)
            # 加载预训练权重的时候，num_classes要用1000，加载完后，我们再单独修改这个fc层中的num_clasees
            # 因为官方给的预训练权重是，按照1000分类给的。

        for m in self.modules():  # 初始化
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, block_first_channel, block_num, stride=1):
        downsample = None

        # 如果步距不为1，就说明进行了下采样，我们就需要定义一个downsample，来让走shortcut的特征图尺寸与走卷积层的尺寸相同。
        if stride != 1 or self.in_channel != block_first_channel * block.expansion:
            # downsample就是shortcut，stride=1，就是实线shortcut，stride=2，就是虚线shortcut。
            # 在18，34层的resnet中，一个block中的通道数是没有发生变化的，所以shortcut中的通道数也是没有发生变化的。
            # 在50，101，152层的resnet中，一个block中的通道数是发生变化的，所以shortcut中的通道数也要发生变化，变为原来的四倍。
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, block_first_channel * block.expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(block_first_channel * block.expansion))

        layers = []  # 每个stage中先把第一个，可能会使用下采样的block加进来。
        layers.append(block(self.in_channel,
                            block_first_channel,
                            stride=stride,
                            downsample=downsample))
        self.in_channel = block_first_channel * block.expansion
        # 在50，101，152层的resnet中，经历过一个下采样的block后，通道数就会都变为输入时的四倍。

        for _ in range(1, block_num):
            layers.append(block(block_first_channel * block.expansion, block_first_channel))
            # 经历过第一个下采样的block后，通道数变为了原来的四倍，例如50层的conv2_x：64 -> 256
            # 后边两个block，就需要先用block中的第一个1*1的卷积层，进行降维，再将通道数：256 -> 64

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet18(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet18-f37072fd.pth
    return ResNet(block_18_34, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)


def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(block_18_34, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(block_50_101_152, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(block_50_101_152, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnet152(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet152-394f9c45.pth
    return ResNet(block_50_101_152, [3, 8, 36, 3], num_classes=num_classes, include_top=include_top)

# a=resnet50()
# print(a)
# 可以输出一下网络结构，对比一下自己的是否正确，或者加载权重，试试。
