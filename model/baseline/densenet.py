import torch
import torch.nn as nn
from torchsummary import summary
import torchsummary


# Bottleneck block to make Dense Blocks
class Bottleneck(nn.Module):
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
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.down_sample(x)


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_class=100, low_resolution = None):
        super().__init__()
        self.growth_rate = growth_rate
        self.low_resolution = low_resolution

        inner_channels = 2 * growth_rate

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
            self.features.add_module("transition_layer_{}".format(index), Transition(inner_channels, out_channels))
            inner_channels = out_channels

        self.features.add_module("dense_block{}".format(len(nblocks) - 1), self._make_dense_layers(block, inner_channels, nblocks[len(nblocks)-1]))
        inner_channels += growth_rate * nblocks[len(nblocks) - 1]

        self.features.add_module('bn', nn.BatchNorm2d(inner_channels))
        self.features.add_module('relu', nn.ReLU(inplace=True))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(inner_channels, num_class)

        self.features = nn.Sequential(*self.features)

        # Initializing_weights
        self._initializing_weights()


    def forward(self, x):
        output = self.conv1(x)
        output = self.features(output)
        output = self.avgpool(output)
        output = output.view(output.size()[0], -1)
        output = self.linear(output)
        return output


    def _make_dense_layers(self, block, in_channels, nblocks):

        dense_block = nn.Sequential()

        for index in range(nblocks):
            dense_block.add_module('bottle_neck_layer_{}'.format(index), block(in_channels, self.growth_rate))
            in_channels += self.growth_rate

        return nn.Sequential(*dense_block)

    
    def _initializing_weights(self) :
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
