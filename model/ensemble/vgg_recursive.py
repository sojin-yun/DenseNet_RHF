import torch
import torch.nn as nn
from torch.nn.modules.container import Sequential
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, StepLR

class VGG_ensemble_recursive(nn.Module):
    def __init__(self, model_key, num_classes = 100, device = None, data = 'mini_imagenet'):
        super(VGG_ensemble_recursive, self).__init__()
        if device != None : self.device = device

        self.data = data
        print(self.data)
        if self.data == 'mini_imagenet' or self.data == 'kidney_stone':
            self.select_model = {
                    '16' : {'conv_layers' : [64, 'R', 128, 'R', 256, 256,      'R', 512, 512,      'R', 512, 512, 'R'],      'boundary_layers' : [64, 128, 256, 512, 512]},
                    '19' : {'conv_layers' : [64, 'R', 128, 'R', 256, 256, 256, 'R', 512, 512, 512, 'R', 512, 512, 512, 'R'], 'boundary_layers' : [64, 128, 256, 512, 512]}
                }
        elif self.data == 'cifar100' :
            self.select_model = {
                    '16' : {'conv_layers' : [64, 64, 128, 'R', 256, 256,      'R', 512, 512,      'R', 512, 512, 512     ], 'boundary_layers' : [128, 256, 512]},
                    '19' : {'conv_layers' : [64, 64, 128, 'R', 256, 256, 256, 'R', 512, 512, 512, 'R', 512, 512, 512, 512], 'boundary_layers' : [128, 256, 512]}
                }

        self.features = self._make_layer_conv(conv_layers = self.select_model[model_key]['conv_layers'])
        self.hf_module = HighFrequencyModule(boundary_layers = self.select_model[model_key]['boundary_layers'], data = self.data, device = self.device)
        self.second_hf_module = HighFrequencyModule(boundary_layers = self.select_model[model_key]['boundary_layers'], data = self.data, device = self.device)
        #self.third_hf_module = HighFrequencyModule(boundary_layers = self.select_model[model_key]['boundary_layers'], data = self.data, device = self.device)

        if self.data == 'mini_imagenet' : width = 7
        elif self.data == 'cifar100' : width = 8
        elif self.data == 'kidney_stone' : width = 16

        self.classifier = nn.Sequential(
            nn.Linear(width * width * 512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )
        self.ensemble_classifier = nn.Sequential(
            nn.Linear(width * width * 1536, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes)
        )

        self.ensemble_relu = nn.ReLU(inplace=True)

        self._initializing_weights()
        
        self.optimizer = optim.SGD(self.parameters(), lr = 0.01, momentum = 0.9, weight_decay=0.0015)
        self.loss = nn.CrossEntropyLoss()
        self.boundary_loss = nn.CrossEntropyLoss()
        self.boundary_loss_2 = nn.CrossEntropyLoss()
        self.ensemble_loss = nn.CrossEntropyLoss()
        self.scheduler = StepLR(self.optimizer, step_size=15, gamma=0.5)


    def _make_layer_conv(self, conv_layers):
        
        model = []
        if self.data == 'kidney_stone' :
            input_size = 3
        else :
            input_size = 3

        for conv in conv_layers:
            if conv == 'R':
                model += [BoundaryConv2d(input_size, input_size, kernel_size=3, stride=1, padding = 1)]
            elif conv == 'M':
                model += [nn.MaxPool2d(2, 2)]
            else:
                model += [nn.Conv2d(input_size, conv, kernel_size=3, stride=1, padding = 1), 
                          nn.BatchNorm2d(conv),
                          nn.ReLU(inplace = True)]
                input_size= conv
        
        return nn.Sequential(*model)


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


    def _initializing_weights(self):
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

        for m in self.hf_module.modules() :
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)


    def _get_boundary_location(self):
        boundary_maps = []
        for m in self.modules():
            if isinstance(m, BoundaryConv2d):
                boundary_maps.append(m.boundary)
        return boundary_maps


    def forward(self, x):
        x = self.features(x)
        x_f = x
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        boundary_maps = self._get_boundary_location()
        b = self.hf_module.boundary_forward(boundary_maps)
        b_f = b
        b = b.view(b.size(0), -1)
        b = self.hf_module.boundary_classifier(b)
        second_boundary_maps = self.hf_module._get_boundary_location()
        b_2 = self.second_hf_module.boundary_forward(second_boundary_maps)
        b_f_2 = b_2
        b_2 = b_2.view(b_2.size(0), -1)
        # third_boundary_maps = self.second_hf_module._get_boundary_location()
        # b_3 = self.third_hf_module.boundary_forward(third_boundary_maps)
        # b_f_3 = b_3
        # b_3 = b_3.view(b_3.size(0), -1)
        # for m, n in zip(second_boundary_maps, third_boundary_maps) :
        #     print(m.shape, n.shape)
        ensemble = torch.cat([x_f, b_f, b_f_2], dim = 1)
        ensemble = self.ensemble_relu(ensemble)
        ensemble = self.ensemble_classifier(ensemble.view(ensemble.size(0), -1))
        return x, b, ensemble, b_2


class BoundaryConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 1):
        super(BoundaryConv2d, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size = kernel_size
        self.stride, self.padding = stride, padding
        self.pooling_kernel_size = 2
        self.boundary = None
                
        self.feed_forward = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding = self.padding)
        
        self.batchnorm_relu = nn.Sequential(
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(True)
        )
        
        self.max_pooling = nn.MaxPool2d(self.pooling_kernel_size)
        
        self.up_sampling = nn.Sequential(
            nn.Upsample(scale_factor=self.pooling_kernel_size, mode = 'bilinear', align_corners=False),
            nn.ReLU(True)
        )


    def forward(self, x):

        # get grad-cam from network, which has parameters trained previous epoch

        # first conv_block
        ret_first_forward = self.feed_forward(x)
        ret_first_forward = self.batchnorm_relu(ret_first_forward)
        ret_pooling = self.max_pooling(ret_first_forward)
        
        # get substracted
        ret_upsample = self.up_sampling(ret_pooling)
        self.boundary = torch.abs(ret_first_forward - ret_upsample)

        return ret_pooling

class HighFrequencyModule(nn.Module) :

    def __init__(self, boundary_layers, data, device) :
        super(HighFrequencyModule, self).__init__()
        self.boundary_layers = boundary_layers
        self.data = data
        self.device = device
        
        if self.data == 'mini_imagenet' : 
            width = 7
            num_classes = 100
        elif self.data == 'cifar100' : 
            width = 8
            num_classes = 100
        elif self.data == 'kidney_stone' : 
            width = 16
            num_classes = 2

        self.boundary_features, self.compression_conv = self._make_boundary_conv(boundary_layers = self.boundary_layers)
        self.boundary_features, self.compression_conv = nn.ModuleList(self.boundary_features), nn.ModuleList(self.compression_conv)
        for m in self.boundary_features : m = m.to(self.device)
        for m in self.compression_conv : m = m.to(self.device)

        self.boundary_classifier = nn.Sequential(
            nn.Linear(width * width * 512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def _make_boundary_conv(self, boundary_layers):
        
        model = []
        comp = []

        for conv in boundary_layers:
            model += [BoundaryFeatures(conv)]
        
        for i in range(len(boundary_layers)-1):
            comp += [nn.Sequential(
                          nn.Conv2d(boundary_layers[i]+boundary_layers[i+1], boundary_layers[i+1], kernel_size=1, stride=1, padding=0),
                          nn.BatchNorm2d(boundary_layers[i+1]),
                          nn.ReLU(inplace = True))]

        return model, comp

    def _get_boundary_location(self):

        boundary_maps = []
        for m in self.modules():
            if isinstance(m, BoundaryFeatures):
                boundary_maps.append(m.boundary)
        return boundary_maps

    def boundary_forward(self, boundary_maps):
        x = None
        for idx in range(len(self.boundary_features)):
            if x is None : 
                x = self.boundary_features[idx](boundary_maps[idx].to(self.device))
            else :
                x = torch.cat([x, boundary_maps[idx].to(self.device)], dim = 1)
                x = self.compression_conv[idx-1](x)
                x = self.boundary_features[idx](x)
        return x

class BoundaryFeatures(nn.Module) :

    def __init__(self, layer) :
        super(BoundaryFeatures, self).__init__()

        self.pooling_kernel_size = 2
        self.conv = nn.Conv2d(layer, layer, kernel_size=5, stride=1, padding = 2)
        self.bn = nn.BatchNorm2d(layer)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d((2, 2))
        self.up_sampling = nn.Sequential(
            nn.Upsample(scale_factor=self.pooling_kernel_size, mode = 'bilinear', align_corners=False),
            nn.ReLU(True)
        )
        self.boundary = None

    def forward(self, x):

        # first conv_block
        ret_first_forward = self.conv(x)
        ret_first_forward = self.bn(ret_first_forward)
        ret_first_forward = self.relu(ret_first_forward)
        ret_pooling = self.maxpool(ret_first_forward)
        
        # get substracted
        ret_upsample = self.up_sampling(ret_pooling)
        self.boundary = torch.abs(ret_first_forward - ret_upsample)

        return ret_pooling