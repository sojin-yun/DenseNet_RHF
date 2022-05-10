import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR
from torchsummary import summary
import torchsummary

class BasicBlock_Rense_Ensemble(nn.Module):

    def __init__(self, in_channels, growth_rate, stride=1):
        super(BasicBlock_Rense_Ensemble, self).__init__()

        out_channels = growth_rate

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        #shortcut
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return torch.cat([x, nn.ReLU()(self.residual_function(x) + self.shortcut(x))], 1)

class RenseNet_ensemble(nn.Module):
    
    def __init__(self, block, num_block, boundary_layers, growth_rate=12, compression=0.5, num_class=2, device = None):
        super(RenseNet_ensemble, self).__init__()

        self.growth_rate = growth_rate
        self.compression = compression
        self.inner_channels = 2 * growth_rate # 24

        if device != None : self.device = device

        self.conv1 = nn.Conv2d(3, 24, kernel_size=7, padding=3, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.relu1 = nn.ReLU(inplace=True)

        self.rense1 = self._make_rense_layers(block, num_block[0])
        self.transit1 = self._make_transit_layers()
        self.rense2 = self._make_rense_layers(block, num_block[1])
        self.transit2 = self._make_transit_layers()
        self.rense3 = self._make_rense_layers(block, num_block[2])
        self.transit3 = self._make_transit_layers()
        self.rense4 = self._make_rense_layers(block, num_block[3])

        self.catch = nn.Identity() #양수 음수 상관없이 다 통과

        self.glob_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Sequential( 
            nn.Linear(self.inner_channels, int(self.inner_channels/2)), nn.ReLU(), 
            nn.Linear(int(self.inner_channels/2), num_class)
        )

        self.boundary_features, self.compression_conv = self._make_boundary_conv(boundary_layers = boundary_layers)
        self.boundary_features, self.compression_conv = nn.ModuleList(self.boundary_features), nn.ModuleList(self.compression_conv)

        boundary_inner_channels = boundary_layers[2]
        self.boundary_fc = nn.Linear(boundary_inner_channels, num_class)
        self.ensemble_fc = nn.Linear(self.inner_channels + boundary_inner_channels, num_class)

        self.ensemble_relu = nn.Identity()

        #self.optimizer = optim.SGD(self.parameters(), lr = 1e-3, momentum = 0.9, weight_decay=0.00001)
        self.optimizer = optim.SGD(self.parameters(), lr = 1e-2, momentum = 0.9, weight_decay=0.0001)
        self.loss = nn.CrossEntropyLoss()
        self.boundary_loss = nn.CrossEntropyLoss()
        self.ensemble_loss = nn.CrossEntropyLoss()
        #self.scheduler = MultiStepLR(self.optimizer, milestones=[100, 150], gamma=0.1)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max = 50, eta_min = 0)

        # Initialization weights
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


    def _make_rense_layers(self, block, nblocks):

        rense_block = nn.Sequential()

        for index in range(nblocks):
            rense_block.add_module('basic_block_layer_{}'.format(index), block(self.inner_channels, self.growth_rate))
            self.inner_channels += self.growth_rate

        return rense_block
    

    def _make_transit_layers(self):

        out_channels = int(self.compression * self.inner_channels)

        downsample = Transition_Rense_Ensemble(self.inner_channels, out_channels)
        
        self.inner_channels = out_channels

        return downsample

    
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
                          nn.MaxPool2d((2, 2), stride = 2))
                          ]
        
        for i in range(len(boundary_layers)-1):
            comp += [nn.Sequential(
                            nn.Conv2d(boundary_layers[i]+boundary_layers[i+1], boundary_layers[i+1], kernel_size=1, stride=1, padding=0),
                            nn.BatchNorm2d(boundary_layers[i+1]),
                            nn.ReLU(inplace = True))]

        return model, comp


    def _get_boundary_location(self):
        boundary_maps = []
        for m in self.modules():
            if isinstance(m, Transition_Rense_Ensemble):
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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.rense1(x)
        x = self.transit1(x)
        x = self.rense2(x)
        x = self.transit2(x)
        x = self.rense3(x)
        x = self.transit3(x)
        x = self.rense4(x)
        x_f = x

        x = self.glob_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        self.boundary_maps = self._get_boundary_location()

        b = self.boundary_forward()

        b_f = b
        b = self.glob_pool(b)
        b = b.view(b.size(0), -1)
        b = self.boundary_fc(b)

        ensemble = torch.cat([x_f, b_f], dim = 1)
        ensemble = self.ensemble_relu(ensemble)

        ensemble = self.glob_pool(ensemble)
        ensemble = ensemble.view(ensemble.size(0), -1)
        ensemble = self.ensemble_fc(ensemble)

        return x, b, ensemble 

class Transition_Rense_Ensemble(nn.Module):

    def __init__(self, in_channels, out_channels) :

        super(Transition_Rense_Ensemble, self).__init__()

        self.batch_norm = nn.BatchNorm2d(in_channels)
        self.conv_1x1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.avg_pool = nn.AvgPool2d(2, stride=2)

        self.up_sampling = nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=False)

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