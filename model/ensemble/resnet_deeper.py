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


class BasicBlock_ensemble_deeper(nn.Module):
    def __init__(self, inplanes: int, midplanes: int, planes: int, stride: int = 1, downsample = None, groups: int = 1, dilation: int = 1, norm_layer = None) :
        super(BasicBlock_ensemble_deeper, self).__init__()
        
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
        self.boundary = None

        # for boundary recovering
        if self.stride == 2:
            self.conv1_strided = conv1x1(inplanes, midplanes)
            self.bn1_strided = norm_layer(midplanes)
            self.upsample = nn.Upsample(scale_factor=self.stride, mode = 'bilinear', align_corners=False)
    

    def _copy_weight(self) :
        with torch.no_grad() :
            self.conv1_strided.weight = self.conv1.weight
            self.conv1_strided.bias = self.conv1.bias


    def forward(self, x) :
        identity = x

        out = self.conv1(x)

        # save boundary
        
        out = self.bn1(out)
        out = self.relu(out)

        if self.stride == 2:
            out_strided = out
            self._copy_weight()
            
            with torch.no_grad() :
                out_no_strided = self.conv1_strided(x)

            out_no_strided = self.bn1_strided(out_no_strided)
            out_no_strided = self.relu(out_no_strided)
            
            out_strided = self.upsample(out_strided)
            out_strided = self.relu(out_strided)
            self.boundary = torch.abs(out_no_strided - out_strided)

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

class ResNet_ensemble_deeper(nn.Module):
    def __init__(self, block, layers, boundary_layers, num_classes = 100, norm_layer = None, device = None, low_resolution = False) :
        super(ResNet_ensemble_deeper, self).__init__()

        self.device = device

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
        self.last_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.fc = nn.Linear(2048, num_classes)

        self.boundary_features, self.compression_conv = self._make_boundary_conv(boundary_layers = boundary_layers)
        self.boundary_features, self.compression_conv = nn.ModuleList(self.boundary_features), nn.ModuleList(self.compression_conv)
        self.boundary_fc = nn.Linear(512, num_classes)
        self.ensemble_fc = nn.Linear(2560, num_classes)

        self.ensemble_relu = nn.ReLU()

        self.optimizer = optim.SGD(self.parameters(), lr = 1e-2, momentum = 0.9, weight_decay=0.0015)
        self.loss = nn.CrossEntropyLoss()
        self.boundary_loss = nn.CrossEntropyLoss()
        self.ensemble_loss = nn.CrossEntropyLoss()
        self.scheduler = StepLR(self.optimizer, step_size=15, gamma=0.5)
        #self.scheduler = MultiStepLR(self.optimizer, milestones=[1, 2, 3], gamma=0.5)

        # weight initialization
        self._initializing_weight()


    def _initializing_weight(self) :

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

        for block in self.boundary_features:
            for module in block:
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight)

        for m in self.compression_conv:
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

    def _make_boundary_conv(self, boundary_layers):
        
        model = []
        comp = []

        for conv in boundary_layers:
            model += [nn.Sequential(
                          # Strided-conv in ResNet structure
                          nn.Conv2d(conv, conv, kernel_size=5, stride=2, padding = 2), 
                          nn.BatchNorm2d(conv),
                          nn.ReLU(inplace = True),)
                          #nn.MaxPool2d((2, 2)))
                          ]
        
        for i in range(len(boundary_layers)-1):
            comp += [nn.Sequential(
                          nn.Conv2d(boundary_layers[i]+boundary_layers[i+1], boundary_layers[i+1], kernel_size=1, stride=1, padding=0),
                          nn.BatchNorm2d(boundary_layers[i+1]),
                          nn.ReLU(inplace = True))]

        return model, comp

    def boundary_forward(self):
        x = None
        for idx in range(len(self.boundary_features)):
            if x is None : 
                x = self.boundary_features[idx](self.boundary_maps[idx].to(self.device))
            else :
                x = torch.cat([x, self.boundary_maps[idx].to(self.device)], dim = 1)
                x = self.compression_conv[idx-1](x)
                x = self.boundary_features[idx](x)
        return x

    def _get_boundary_location(self):
        boundary_maps = []
        for m in self.modules():
            if isinstance(m, BasicBlock_ensemble_deeper):
                if m.stride == 2:
                    boundary_maps.append(m.boundary)
        return boundary_maps

    def forward(self, x):

        x = self.former_block(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x_f = x
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        self.boundary_maps = self._get_boundary_location()

        b = self.boundary_forward()

        b_f = b
        b = self.avgpool(b)
        b = b.view(b.size(0), -1)
        b = self.boundary_fc(b)

        ensemble = torch.cat([x_f, b_f], dim = 1)
        ensemble = self.ensemble_relu(ensemble)

        ensemble = self.avgpool(ensemble)
        ensemble = ensemble.view(ensemble.size(0), -1)
        ensemble = self.ensemble_fc(ensemble)

        return x, b, ensemble