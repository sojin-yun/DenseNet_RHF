import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from utils.module import ReviveConv2d, ReviveAvgPool2d, ReviveMaxPool2d

class Fire_ensemble(nn.Module):
    def __init__(self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int) -> None:
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat(
            [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        )


class SqueezeNet_ensemble(nn.Module):
    def __init__(self, version: str = "1_0", num_classes: int = 100, dropout: float = 0.5, device = None) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        if version == "1_0":
            self.features = nn.Sequential(
                ReviveConv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                ReviveMaxPool2d(kernel_size=3, stride=2, ceil_mode=True, upsample_size=109),
                Fire_ensemble(96, 16, 64, 64),
                Fire_ensemble(128, 16, 64, 64),
                Fire_ensemble(128, 32, 128, 128),
                ReviveMaxPool2d(kernel_size=3, stride=2, ceil_mode=True, upsample_size=54),
                Fire_ensemble(256, 32, 128, 128),
                Fire_ensemble(256, 48, 192, 192),
                Fire_ensemble(384, 48, 192, 192),
                Fire_ensemble(384, 64, 256, 256),
                ReviveMaxPool2d(kernel_size=3, stride=2, ceil_mode=True, upsample_size=27),
                Fire_ensemble(512, 64, 256, 256),
            )
            added_channel = 512
            boundary_layers = [96, 96, 256, 512]
        elif version == "1_1":
            self.features = nn.Sequential(
                ReviveConv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                ReviveMaxPool2d(kernel_size=3, stride=2, ceil_mode=True, upsample_size=111),
                Fire_ensemble(64, 16, 64, 64),
                Fire_ensemble(128, 16, 64, 64),
                ReviveMaxPool2d(kernel_size=3, stride=2, ceil_mode=True, upsample_size=55),
                Fire_ensemble(128, 32, 128, 128),
                Fire_ensemble(256, 32, 128, 128),
                ReviveMaxPool2d(kernel_size=3, stride=2, ceil_mode=True, upsample_size=27),
                Fire_ensemble(256, 48, 192, 192),
                Fire_ensemble(384, 48, 192, 192),
                Fire_ensemble(384, 64, 256, 256),
                Fire_ensemble(512, 64, 256, 256),
            )
            added_channel = 256
            boundary_layers = [64, 64, 128, 256]
        else:
            # FIXME: Is this needed? SqueezeNet should only be called from the
            # FIXME: squeezenet1_x() functions
            # FIXME: This checking is not done for the other models
            raise ValueError(f"Unsupported SqueezeNet version {version}: 1_0 or 1_1 expected")

        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        boundary_final_conv = nn.Conv2d(added_channel, self.num_classes, kernel_size=1)
        ensemble_final_conv = nn.Conv2d(512+added_channel, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential( nn.Dropout(p=dropout), final_conv, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)))
        self.boundary_classifier = nn.Sequential(boundary_final_conv, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)))
        self.ensemble_classifier = nn.Sequential(ensemble_final_conv, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)))
        self.boundary_features, self.compression_conv = self._make_boundary_conv(boundary_layers = boundary_layers)
        self.boundary_features, self.compression_conv = nn.ModuleList(self.boundary_features), nn.ModuleList(self.compression_conv)
        
        for m in self.boundary_features : 
            m = m.to(self.device)
        for m in self.compression_conv : 
            m = m.to(self.device)
        
        self.optimizer = optim.SGD(self.parameters(), lr = 1e-2, momentum = 0.9, weight_decay=0.0015)
        self.loss = nn.CrossEntropyLoss()
        self.boundary_loss = nn.CrossEntropyLoss()
        self.ensemble_loss = nn.CrossEntropyLoss()
        #self.scheduler = StepLR(self.optimizer, step_size=15, gamma=0.5)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max = 10, eta_min = 0)
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
                    
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
                    
    def _make_boundary_conv(self, boundary_layers):
        
        model = []
        comp = []

        for conv in boundary_layers:
            model += [nn.Sequential(
                          nn.Conv2d(conv, conv, kernel_size=3, stride = 1, padding = 1, groups = conv), 
                          nn.BatchNorm2d(conv),
                          nn.LeakyReLU(inplace = True),
                          nn.Conv2d(conv, conv, kernel_size=1, stride = 1, padding = 0), 
                          nn.BatchNorm2d(conv),
                          nn.LeakyReLU(inplace = True),
                          nn.MaxPool2d((2, 2)))]
        
        for i in range(len(boundary_layers)-1):
            comp += [nn.Sequential(
                          nn.Conv2d(boundary_layers[i]+boundary_layers[i+1], boundary_layers[i+1], kernel_size=1, stride=1, padding=0),
                          nn.BatchNorm2d(boundary_layers[i+1]),
                          nn.LeakyReLU(inplace = True))]

        return model, comp
    
    def _get_boundary_location(self):
        boundary_maps = []
        for m in self.modules():
            if isinstance(m, ReviveConv2d) or isinstance(m, ReviveAvgPool2d) or isinstance(m, ReviveMaxPool2d):
                boundary_maps.append(m.recovered_map)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x_f = x
        x = self.classifier(x)
        self.boundary_maps = self._get_boundary_location()
        b = self.boundary_forward()
        b_f = b
        b = self.boundary_classifier(b)
        out = torch.cat([x_f, b_f], dim = 1)
        out = self.ensemble_classifier(out)
        x, b, out = torch.flatten(x, 1), torch.flatten(b, 1), torch.flatten(out, 1)
        return x, b, out


def _squeezenet_ensemble(version: str, device, num_classes) -> SqueezeNet_ensemble:
    model = SqueezeNet_ensemble(version, device = device, num_classes = num_classes)
    return model