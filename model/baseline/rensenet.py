import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR
from torchsummary import summary
import torchsummary

class BasicBlock_Rense(nn.Module):

    def __init__(self, in_channels, growth_rate, stride=1):
        super(BasicBlock_Rense, self).__init__()

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

class RenseNet(nn.Module):
    
    def __init__(self, block, num_block, growth_rate=12, compression=0.5, num_class=2, device = None):
        super(RenseNet, self).__init__()

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

        self.ensemble_relu = nn.Identity() #양수 음수 상관없이 다 통과

        self.glob_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Sequential( 
            nn.Linear(self.inner_channels, int(self.inner_channels/2)), nn.ReLU(), 
            nn.Linear(int(self.inner_channels/2), num_class)
        )

        #self.optimizer = optim.SGD(self.parameters(), lr = 1e-3, momentum = 0.9, weight_decay=0.00001)
        self.optimizer = optim.SGD(self.parameters(), lr = 1e-2, momentum = 0.9, weight_decay=0.0001)
        self.loss = nn.CrossEntropyLoss()
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


    def _make_rense_layers(self, block, nblocks):

        rense_block = nn.Sequential()

        for index in range(nblocks):
            rense_block.add_module('basic_block_layer_{}'.format(index), block(self.inner_channels, self.growth_rate))
            self.inner_channels += self.growth_rate

        return rense_block
    

    def _make_transit_layers(self):

        out_channels = int(self.compression * self.inner_channels)

        downsample = Transition_Rense(self.inner_channels, out_channels)
        
        self.inner_channels = out_channels

        return downsample
    
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
        x = self.ensemble_relu(x)

        x = self.glob_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x

class Transition_Rense(nn.Module):

    def __init__(self, in_channels, out_channels) :

        super(Transition_Rense, self).__init__()

        self.batch_norm = nn.BatchNorm2d(in_channels)
        self.conv_1x1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.avg_pool = nn.AvgPool2d(2, stride=2)

    def forward(self, x):

        # Transition block's feed forward
        x = self.batch_norm(x)
        x = self.conv_1x1(x)

        # AvgPool2d, following forward path
        x = self.avg_pool(x)

        return x