import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Swish activation function
class Swish_ensemble(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


# SE Block
class SEBlock_ensemble(nn.Module):
    def __init__(self, in_channels, r=4):
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool2d((1,1))
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels * r),
            Swish_ensemble(),
            nn.Linear(in_channels * r, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.squeeze(x)
        x = x.view(x.size(0), -1)
        x = self.excitation(x)
        x = x.view(x.size(0), x.size(1), 1, 1)
        return x


class MBConv_ensemble(nn.Module):
    expand = 6
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, se_scale=4, p=0.5):
        super().__init__()
        # first MBConv is not using stochastic depth
        self.p = torch.tensor(p).float() if (in_channels == out_channels) else torch.tensor(1).float()

        self.strided_conv = nn.Conv2d(in_channels, in_channels * MBConv_ensemble.expand, 1, stride=stride, padding=0, bias=False)
        self.strided_bn = nn.BatchNorm2d(in_channels * MBConv_ensemble.expand, momentum=0.99, eps=1e-3)
        self.strided_swish = Swish_ensemble()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels * MBConv_ensemble.expand, in_channels * MBConv_ensemble.expand, kernel_size=kernel_size,
                      stride=1, padding=kernel_size//2, bias=False, groups=in_channels*MBConv_ensemble.expand),
            nn.BatchNorm2d(in_channels * MBConv_ensemble.expand, momentum=0.99, eps=1e-3),
            Swish_ensemble()
        )

        self.se = SEBlock_ensemble(in_channels * MBConv_ensemble.expand, se_scale)

        self.project = nn.Sequential(
            nn.Conv2d(in_channels*MBConv_ensemble.expand, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3)
        )

        self.shortcut = (stride == 1) and (in_channels == out_channels)

        self.boundary = None
        self.stride = stride

        if self.stride == 2 :
            self.ext_conv = nn.Conv2d(in_channels, in_channels * MBConv_ensemble.expand, 1, stride=1, padding=0, bias=False)
            self.ext_bn = nn.BatchNorm2d(in_channels * MBConv_ensemble.expand, momentum=0.99, eps=1e-3)
            self.ext_swish = Swish_ensemble()
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=self.stride, mode = 'bilinear', align_corners=False)
            )
            self.relu = nn.ReLU()

    def forward(self, x):
        # stochastic depth
        if self.training:
            if not torch.bernoulli(self.p):
                return x

        x_shortcut = x
        x_org = x

        x = self.strided_conv(x)
        x = self.strided_bn(x)
        x = self.strided_swish(x)

        if self.stride == 2 :
            x_stride = x
            self._copy_weight()

            with torch.no_grad() :
                x_no_strided = self.ext_conv(x_org)

            x_no_strided = self.ext_bn(x_no_strided)
            x_no_strided = self.ext_swish(x_no_strided)
            
            x_stride = self.upsample(x_stride)
            x_stride = self.relu(x_stride)
            self.boundary = torch.abs(x_no_strided - x_stride)

        x_residual = self.residual(x)
        x_se = self.se(x_residual)

        x = x_se * x_residual
        x = self.project(x)

        if self.shortcut:
            x= x_shortcut + x

        return x

    def _copy_weight(self) :
        with torch.no_grad() :
            self.ext_conv.weight = self.strided_conv.weight
            self.ext_conv.bias = self.strided_conv.bias



class SepConv_ensemble(nn.Module):
    expand = 1
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, se_scale=4, p=0.5):
        super().__init__()
        # first SepConv is not using stochastic depth
        self.p = torch.tensor(p).float() if (in_channels == out_channels) else torch.tensor(1).float()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels * SepConv_ensemble.expand, in_channels * SepConv_ensemble.expand, kernel_size=kernel_size,
                      stride=1, padding=kernel_size//2, bias=False, groups=in_channels*SepConv_ensemble.expand),
            nn.BatchNorm2d(in_channels * SepConv_ensemble.expand, momentum=0.99, eps=1e-3),
            Swish_ensemble()
        )

        self.se = SEBlock_ensemble(in_channels * SepConv_ensemble.expand, se_scale)

        self.project = nn.Sequential(
            nn.Conv2d(in_channels*SepConv_ensemble.expand, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3)
        )

        self.shortcut = (stride == 1) and (in_channels == out_channels)

    def forward(self, x):
        # stochastic depth
        if self.training:
            if not torch.bernoulli(self.p):
                return x

        x_shortcut = x
        x_residual = self.residual(x)
        x_se = self.se(x_residual)

        x = x_se * x_residual
        x = self.project(x)

        if self.shortcut:
            x= x_shortcut + x

        return x


class EfficientNet_ensemble(nn.Module):
    def __init__(self, boundary_layers, num_classes=10, width_coef=1., depth_coef=1., scale=1., dropout=0.2, se_scale=4, stochastic_depth=False, p=0.5, device = None):
        super().__init__()
        channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        repeats = [1, 2, 2, 3, 3, 4, 1]
        strides = [1, 2, 2, 2, 1, 2, 1]
        kernel_size = [3, 3, 5, 3, 5, 5, 3]
        depth = depth_coef
        width = width_coef

        channels = [int(x*width) for x in channels]
        repeats = [int(x*depth) for x in repeats]

        # stochastic depth
        if stochastic_depth:
            self.p = p
            self.step = (1 - 0.5) / (sum(repeats) - 1)
        else:
            self.p = 1
            self.step = 0


        # efficient net
        self.upsample = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False)

        self.stage1 = nn.Sequential(
            nn.Conv2d(3, channels[0],3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0], momentum=0.99, eps=1e-3)
        )

        self.stage2 = self._make_Block(SepConv_ensemble, repeats[0], channels[0], channels[1], kernel_size[0], strides[0], se_scale)

        self.stage3 = self._make_Block(MBConv_ensemble, repeats[1], channels[1], channels[2], kernel_size[1], strides[1], se_scale)

        self.stage4 = self._make_Block(MBConv_ensemble, repeats[2], channels[2], channels[3], kernel_size[2], strides[2], se_scale)

        self.stage5 = self._make_Block(MBConv_ensemble, repeats[3], channels[3], channels[4], kernel_size[3], strides[3], se_scale)

        self.stage6 = self._make_Block(MBConv_ensemble, repeats[4], channels[4], channels[5], kernel_size[4], strides[4], se_scale)

        self.stage7 = self._make_Block(MBConv_ensemble, repeats[5], channels[5], channels[6], kernel_size[5], strides[5], se_scale)

        self.stage8 = self._make_Block(MBConv_ensemble, repeats[6], channels[6], channels[7], kernel_size[6], strides[6], se_scale)

        self.stage9 = nn.Sequential(
            nn.Conv2d(channels[7], channels[8], 1, stride=1, bias=False),
            nn.BatchNorm2d(channels[8], momentum=0.99, eps=1e-3),
            Swish_ensemble()
        ) 

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(channels[8], num_classes)

        self.device = device

        self.boundary_features, self.compression_conv = self._make_boundary_conv(boundary_layers = boundary_layers)
        self.boundary_features, self.compression_conv = nn.ModuleList(self.boundary_features), nn.ModuleList(self.compression_conv)

        self.ensemble_relu = nn.Identity()

        self.optimizer = optim.SGD(self.parameters(), lr = 1e-2, momentum = 0.9, weight_decay=0.0001)
        self.loss = nn.CrossEntropyLoss()
        self.boundary_loss = nn.CrossEntropyLoss()
        self.ensemble_loss = nn.CrossEntropyLoss()
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max = 50, eta_min = 0)

        self._initializing_weights()

    def _initializing_weights(self) :

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
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

    def forward(self, x):
        x = self.upsample(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.stage8(x)
        x = self.stage9(x)
        x_f = x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.linear(x)

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

        return x


    def _make_Block(self, block, repeats, in_channels, out_channels, kernel_size, stride, se_scale):
        strides = [stride] + [1] * (repeats - 1)
        layers = []
        for stride in strides:
            layers.append(block(in_channels, out_channels, kernel_size, stride, se_scale, self.p))
            in_channels = out_channels
            self.p -= self.step

        return nn.Sequential(*layers)

    def _get_boundary_location(self):
        boundary_maps = []
        for m in self.modules():
            if isinstance(m, MBConv_ensemble):
                if m.stride == 2:
                    print(m.boundary.shape)
                    boundary_maps.append(m.boundary)
        return boundary_maps

    def _make_boundary_conv(self, boundary_layers):
        
        model = []
        comp = []

        for conv in boundary_layers:
            model += [nn.Sequential(
                          # Strided-conv in ResNet structure
                          nn.Conv2d(conv, conv, kernel_size=5, stride=1, padding = 2), 
                          nn.BatchNorm2d(conv),
                          nn.LeakyReLU(inplace = True),
                          nn.Conv2d(conv, conv, kernel_size=5, stride=1, padding = 2), 
                          nn.BatchNorm2d(conv),
                          nn.LeakyReLU(inplace = True),
                          nn.MaxPool2d((2, 2)))
                          ]
        
        for i in range(len(boundary_layers)-1):
            comp += [nn.Sequential(
                          nn.Conv2d(boundary_layers[i]+boundary_layers[i+1], boundary_layers[i+1], kernel_size=1, stride=1, padding=0),
                          nn.BatchNorm2d(boundary_layers[i+1]),
                          nn.LeakyReLU(inplace = True))]

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

