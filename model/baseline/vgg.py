import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, StepLR

class VGG(nn.Module):

    def __init__(self, model_key, num_classes = 100, data = 'mini_imagenet'):
        super().__init__()


        if data == 'mini_imagenet' :
            self.select_model = {
                    '16' : {'conv_layers' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512, 'M']},
                    '19' : {'conv_layers' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']}
                }
        elif data == 'cifar100' :
            self.select_model = {
                    '16' : {'conv_layers' : [64, 64, 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512]},
                    '19' : {'conv_layers' : [64, 64, 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]}
                }

        self.features = self._make_layers(self.select_model[model_key]['conv_layers'])

        if data == 'mini_imagenet' : width = 7
        elif data == 'cifar100' : width = 8

        self.classifier = nn.Sequential(
            nn.Linear(width * width* 512, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        )

        self.ensemble_relu = nn.ReLU(inplace=True)

        self._initialize_weights()
        
        self.optimizer = optim.SGD(self.parameters(), lr = 0.01, momentum = 0.9, weight_decay=0.0015)
        self.loss = nn.CrossEntropyLoss()
        self.scheduler = StepLR(self.optimizer, step_size=15, gamma=0.5)


    def _make_layers(self, select_model):
        layers = []

        input_channel = 3
        for l in select_model:
            if l == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                continue

            layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1),
                       nn.BatchNorm2d(l),
                       nn.ReLU(inplace=True)]
            input_channel = l

        return nn.Sequential(*layers)


    def _initialize_weights(self) :
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
    
    
    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size(0), -1)
        output = self.classifier(output)

        return output