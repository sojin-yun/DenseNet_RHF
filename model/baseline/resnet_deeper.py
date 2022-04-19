import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, StepLR


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) :
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) :
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock_deeper(nn.Module):
    def __init__(self, inplanes: int, midplanes: int, planes: int, stride: int = 1, downsample = None, groups: int = 1, dilation: int = 1, norm_layer = None) :
        super(BasicBlock_deeper, self).__init__()
        
        # Normalization Layer
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv1x1(inplanes, midplanes, stride)
        self.bn1 = norm_layer(midplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(midplanes, midplanes)
        self.bn2 = norm_layer(midplanes)
        self.conv3 = conv1x1(midplanes, planes)
        self.bn3 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x) :
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        # downsampling이 필요한 경우 downsample layer를 block에 인자로 넣어주어야함
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # residual connection
        out = self.relu(out)

        return out

class ResNet_deeper(nn.Module):
    def __init__(self, block, layers, num_classes = 100, norm_layer = None, low_resolution = False) :
        super(ResNet_deeper, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer  # batch norm layer

        self.inplanes = 64  # input shape
        self.dilation = 1  # dilation fixed
        self.groups = 1  # groups fixed
        
        # input block
        if not low_resolution :
            self.former_block = nn.Sequential(
                nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
                norm_layer(self.inplanes),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else :
            self.former_block = nn.Sequential(
                nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(self.inplanes),
                nn.ReLU(inplace=True)
            )
        
        # residual blocks
        self.layer1 = self._make_layer(block, 64, 64, 256, layers[0])
        self.layer2 = self._make_layer(block, 256, 128, 512, layers[1], stride=2, dilate=False)
        self.layer3 = self._make_layer(block, 512, 256, 1024, layers[2], stride=2, dilate=False)
        self.layer4 = self._make_layer(block, 1024, 512, 2048, layers[3], stride=2, dilate=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

        self.cam_relu = nn.Identity()

        #self.optimizer = optim.SGD(self.parameters(), lr = 1e-3, momentum = 0.9, weight_decay=0.00001)
        self.optimizer = optim.SGD(self.parameters(), lr = 1e-2, momentum = 0.9, weight_decay=0.0001)
        self.loss = nn.CrossEntropyLoss()
        #self.scheduler = StepLR(self.optimizer, step_size=15, gamma=0.5)
        self.scheduler = MultiStepLR(self.optimizer, milestones=[60, 90], gamma=0.1)

        # weight initialization
        self._initializing_weights()


    def _initializing_weights(self) :
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                

    def _make_layer(self, block, inplanes: int, midplanes:int, planes: int, blocks: int, stride: int = 1, dilate: bool = False):
        norm_layer = self._norm_layer
        downsample = None
        
        # downsampling 필요할경우 downsample layer 생성
        if stride != 1 or self.inplanes != planes:  
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                norm_layer(planes),
            )

        layers = []
        layers.append(block(inplanes, midplanes, planes, stride, downsample, self.groups, self.dilation, norm_layer))
        
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, midplanes, planes, groups=self.groups, dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)


    def forward(self, x):

        print(x.shape)

        x = self.former_block(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.cam_relu(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        #x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x