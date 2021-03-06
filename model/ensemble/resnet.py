import torch
from torch._C import device
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from torchsummary import summary

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) :
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) :
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock_ensemble(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample = None, groups: int = 1, dilation: int = 1, norm_layer = None) :
        super(BasicBlock_ensemble, self).__init__()
        
        # Normalization Layer
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.boundary = None

        # for boundary recovering
        if self.stride == 2:
            self.conv1_strided = conv3x3(inplanes, planes)
            self.bn1_strided = norm_layer(planes)
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

        
        # downsampling??? ????????? ?????? downsample layer??? block??? ????????? ??????????????????
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # residual connection
        out = self.relu(out)

        return out

class ResNet_ensemble(nn.Module):
    def __init__(self, block, layers, boundary_layers, num_classes = 100, norm_layer = None, device = None) :
        super(ResNet_ensemble, self).__init__()

        self.device = device

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer  # batch norm layer

        self.inplanes = 64  # input shape
        self.dilation = 1  # dilation fixed
        self.groups = 1  # groups fixed
        
        # input block
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # residual blocks
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=False)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=False)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=False)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        self.boundary_features, self.compression_conv = self._make_boundary_conv(boundary_layers = boundary_layers)
        self.boundary_features, self.compression_conv = nn.ModuleList(self.boundary_features), nn.ModuleList(self.compression_conv)
        self.boundary_fc = nn.Linear(512, num_classes)
        self.ensemble_fc = nn.Linear(1024, num_classes)

        self.ensemble_relu = nn.ReLU(inplace=True)

        self.optimizer = optim.SGD(self.parameters(), lr = 1e-2, momentum = 0.9, weight_decay=0.0015)
        #self.optimizer = optim.SGD(self.parameters(), lr = 1e-1, momentum = 0.9, weight_decay=1e-3)
        self.loss = nn.CrossEntropyLoss()
        self.boundary_loss = nn.CrossEntropyLoss()
        self.ensemble_loss = nn.CrossEntropyLoss()
        self.scheduler = StepLR(self.optimizer, step_size=12, gamma=0.5)
        #self.scheduler = MultiStepLR(self.optimizer, milestones=[40, 80, 120, 140, 160], gamma = 0.2)

        # weight initialization
        for idx, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1, dilate: bool = False):
        norm_layer = self._norm_layer
        downsample = None
        
        # downsampling ??????????????? downsample layer ??????
        if stride != 1 or self.inplanes != planes:  
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                norm_layer(planes),
            )

        layers = []
        layers.append(block(inplanes = self.inplanes, planes = planes, stride = stride, downsample = downsample, groups = self.groups, dilation = self.dilation, norm_layer = norm_layer))
        self.inplanes = planes

        for _ in range(1, blocks):
            layers.append(block(inplanes = self.inplanes, planes = planes, groups = self.groups, dilation = self.dilation, norm_layer = norm_layer))

        return nn.Sequential(*layers)

    def _make_boundary_conv(self, boundary_layers):
        
        model = []
        comp = []

        for conv in boundary_layers:
            model += [nn.Sequential(
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
                if self.device == None :
                    x = self.boundary_features[idx](self.boundary_maps[idx].to(torch.device(torch.cuda.current_device())))
                else :
                    x = self.boundary_features[idx](self.boundary_maps[idx].to(self.device))
            else :
                if self.device == None :
                    x = torch.cat([x, self.boundary_maps[idx].to(torch.device(torch.cuda.current_device()))], dim = 1)
                else :
                    x = torch.cat([x, self.boundary_maps[idx].to(self.device)], dim = 1)
                x = self.compression_conv[idx-1](x)
                x = F.relu(x)
                x = self.boundary_features[idx](x)
        return x

    def _get_boundary_location(self):
        boundary_maps = []
        for m in self.modules():
            if isinstance(m, BasicBlock_ensemble):
                if m.stride == 2:
                    boundary_maps.append(m.boundary)
        return boundary_maps

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

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