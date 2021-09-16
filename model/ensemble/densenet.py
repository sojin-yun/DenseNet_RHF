import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
import torchsummary


# Bottleneck block to make Dense Blocks
class Bottleneck_ensemble(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        
        inner_channel = 4 * growth_rate

        self.bottle_neck = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inner_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channel, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return torch.cat([x, self.bottle_neck(x)], 1)


# Transition block to match feature maps in various layers
class Transition_ensemble(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.batch_norm = nn.BatchNorm2d(in_channels)
        self.conv_1x1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.avg_pool = nn.AvgPool2d(2, stride=2)

        self.up_sampling = nn.Sequential(
            nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=False),
            nn.ReLU(True)
        )

        self.boundary = None


    def forward(self, x):

        # Transition block's feed forward
        x = self.batch_norm(x)
        x = self.conv_1x1(x)

        # Activation map before AvgPool2d
        ret_forward = x

        # AvgPool2d, following forward path
        x = self.avg_pool(x)

        # Upsampling to get boundary map
        ret_pool = x
        ret_upsample = self.up_sampling(ret_pool)
        self.boundary = torch.abs(ret_forward - ret_upsample)

        return x


class DenseNet_ensemble(nn.Module):
    def __init__(self, block, nblocks, boundary_layers, growth_rate=12, reduction=0.5, num_class=100, device = None, low_resolution = False):
        super().__init__()
        self.growth_rate = growth_rate
        self.low_resolution = low_resolution

        inner_channels = 2 * growth_rate

        if device != None : self.device = device

        self.features = nn.Sequential()

        if not self.low_resolution :
            self.features.add_module('conv0', nn.Conv2d(3, inner_channels, kernel_size = 7, stride = 2, padding = 3, bias = False))
            self.features.add_module('norm0', nn.BatchNorm2d(inner_channels))
            self.features.add_module('avgpool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            
        else :
            self.features.add_module('conv0', nn.Conv2d(3, inner_channels, kernel_size=3, padding=1, bias=False))
            self.features.add_module('norm0', nn.BatchNorm2d(inner_channels))


        for index in range(len(nblocks) - 1):
            self.features.add_module("dense_block_layer_{}".format(index), self._make_dense_layers(block, inner_channels, nblocks[index]))
            inner_channels += growth_rate * nblocks[index]

            out_channels = int(reduction * inner_channels) 
            self.features.add_module("transition_layer_{}".format(index), Transition_ensemble(inner_channels, out_channels))
            inner_channels = out_channels

        self.features.add_module("dense_block{}".format(len(nblocks) - 1), self._make_dense_layers(block, inner_channels, nblocks[len(nblocks)-1]))
        inner_channels += growth_rate * nblocks[len(nblocks) - 1]

        self.features.add_module('bn', nn.BatchNorm2d(inner_channels))
        self.features.add_module('relu', nn.ReLU(inplace=True))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.features = nn.Sequential(*self.features)

        self.boundary_features, self.compression_conv = self._make_boundary_conv(boundary_layers = boundary_layers)
        self.boundary_features, self.compression_conv = nn.ModuleList(self.boundary_features), nn.ModuleList(self.compression_conv)

        self.fc = nn.Linear(inner_channels, num_class)
        self.boundary_fc = nn.Linear(512, num_class)
        self.ensemble_fc = nn.Linear(inner_channels + 512, num_class)

        self.ensemble_relu = nn.ReLU(inplace=True)

        self.optimizer = optim.SGD(self.parameters(), lr = 1e-2, momentum = 0.9, weight_decay=0.0015)
        self.loss = nn.CrossEntropyLoss()
        self.boundary_loss = nn.CrossEntropyLoss()
        self.ensemble_loss = nn.CrossEntropyLoss()
        self.scheduler = StepLR(self.optimizer, step_size=15, gamma=0.5)

        # Initialization weights
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


    def _make_dense_layers(self, block, in_channels, nblocks):
        dense_block = nn.Sequential()
        for index in range(nblocks):
            dense_block.add_module('bottle_neck_layer_{}'.format(index), block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return nn.Sequential(*dense_block)


    def _make_boundary_conv(self, boundary_layers):
        
        model = []
        comp = []

        for conv in boundary_layers:
            model += [nn.Sequential(
                          nn.Conv2d(conv, conv, kernel_size=5, stride=1, padding = 2), 
                          nn.BatchNorm2d(conv),
                          nn.ReLU(inplace = True),
                          nn.MaxPool2d((2, 2)))]
        
        for i in range(len(boundary_layers)-1):
            comp += [nn.Sequential(
                          nn.Conv2d(boundary_layers[i]+boundary_layers[i+1], boundary_layers[i+1], kernel_size=1, stride=1, padding=0),
                          nn.BatchNorm2d(boundary_layers[i+1]),
                          nn.ReLU(inplace = True))]

        return model, comp


    def _get_boundary_location(self):
        boundary_maps = []
        for m in self.modules():
            if isinstance(m, Transition_ensemble):
                boundary_maps.append(m.boundary)
        return boundary_maps


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


    def forward(self, x):
        x = self.features(x)
        x_f = x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
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
