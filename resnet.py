'''ResNet-34 Image classfication for cifar-10 with PyTorch

Author 'Yao-jia'.

'''

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

def conv3x3(inchannel, outchannel, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    def __init__(self,inchannel,outchannel,stride=1):
        super(ResidualBlock,self).__init__()
        self.conv1 = conv3x3(inchannel,outchannel,stride)
        self.bn1 = nn.BatchNorm2d(outchannel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(outchannel,outchannel)
        self.bn2 = nn.BatchNorm2d(outchannel)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inchannel, outchannel, stride=1):
        super(Bottleneck,self).__init__()
        self.conv1 = nn.Conv2d(inchannel,outchannel,kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm2d(outchannel)
        self.conv2 = nn.Conv2d(outchannel,outchannel,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(outchannel)
        self.conv3 = nn.Conv2d(outchannel,outchannel*4,kernel_size=1,bias=False)
        self.bn3 = nn.BatchNorm2d(outchannel*4)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(x)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        self.inchannel = 64
        super(ResNet,self).__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1])
        self.layer3 = self._make_layer(block, 256, layers[2])
        self.layer4 = self._make_layer(block, 512, layers[3])
        self.avgpool = nn.AvgPool2d(7,stride=1)

        self.fc = nn.Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, outchannel, blocks, stride=1):
        layers = []
        layers.append(block(self.inchannel, outchannel, stride))
        self.inchannel = outchannel * block.expansion
        for i in range(1,blocks):
            layers.append(block(self.inchannel,outchannel))

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

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet34():
    model = ResNet(BasicBlock, [3,4,6,3])
    return model



