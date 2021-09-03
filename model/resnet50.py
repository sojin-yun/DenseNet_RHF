class ResNet_ensemble(nn.Module):
    def __init__(self, block, layers, boundary_layers, num_classes = 55, norm_layer = None, resnet_50 = None, device = None) :
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
        if resnet_50 == None :
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=False)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=False)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=False)
        else :
            self.layer1 = self._make_layer_50(block, 64, 64, 256, layers[0])
            self.layer2 = self._make_layer_50(block, 256, 128, 512, layers[1], stride=2, dilate=False)
            self.layer3 = self._make_layer_50(block, 512, 256, 1024, layers[2], stride=2, dilate=False)
            self.layer4 = self._make_layer_50(block, 1024, 512, 2048, layers[3], stride=2, dilate=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.last_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        #self.fc = nn.Linear(2048, num_classes)
        self.fc = nn.Linear(20480, num_classes)

        self.boundary_features, self.compression_conv = self._make_boundary_conv(boundary_layers = boundary_layers)
        self.boundary_features, self.compression_conv = nn.ModuleList(self.boundary_features), nn.ModuleList(self.compression_conv)
        #self.boundary_fc = nn.Linear(512, num_classes)
        self.boundary_fc = nn.Linear(5120, num_classes)
        #self.ensemble_fc = nn.Linear(2560, num_classes)
        self.ensemble_fc = nn.Linear(25600, num_classes)

        # output_size, width = num_classes, 3

        # self.last_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # self.boundary_last_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        # self.ensemble_last_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        # self.classifier = nn.Sequential(
        #         nn.Linear(width * width * 512 + 512, 1024),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(1024, 512),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(512, output_size)
        # )
        # self.boundary_classifier = nn.Sequential(
        #     #nn.Dropout(0.2),
        #     nn.Linear(width * width * 512 + 512, 1024),
        #     nn.ReLU(inplace=True),
        #     #nn.Dropout(0.2),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(512, output_size)
        # )
        # self.ensemble_classifier = nn.Sequential(
        #     nn.Linear(width * width * 1024  + 1024, 2048),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(2048, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(512, output_size)
        # )

        self.ensemble_relu = nn.ReLU(inplace=True)

        self.optimizer = optim.SGD(self.parameters(), lr = 1e-2, momentum = 0.9, weight_decay=0.0015)
        self.loss = nn.CrossEntropyLoss()
        self.boundary_loss = nn.CrossEntropyLoss()
        self.ensemble_loss = nn.CrossEntropyLoss()
        self.scheduler = StepLR(self.optimizer, step_size=12, gamma=0.5)
        #self.scheduler = MultiStepLR(self.optimizer, milestones=[1, 2, 3], gamma=0.5)

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
        
        # downsampling 필요할경우 downsample layer 생성
        if stride != 1 or self.inplanes != planes:  
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                norm_layer(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.dilation, norm_layer))
        self.inplanes = planes

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.groups, self.dilation, norm_layer))

        return nn.Sequential(*layers)

    def _make_layer_50(self, block, inplanes: int, midplanes:int, planes: int, blocks: int, stride: int = 1, dilate: bool = False):
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
                          nn.Conv2d(conv, conv, kernel_size=5, stride=2, padding = 2), 
                          nn.BatchNorm2d(conv),
                          nn.ReLU(inplace = True),)
                          #nn.MaxPool2d((2, 2)))
                          ]
        
        for i in range(len(boundary_layers)-1):
            comp += [nn.Conv2d(boundary_layers[i]+boundary_layers[i+1], boundary_layers[i+1], kernel_size=1, stride=1, padding=0)]

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
            if isinstance(m, BasicBlock_ensemble_50):
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

        x_flatten = self.last_pool(x)
        x_flatten = x_flatten.view(x_flatten.size(0), -1)

        x_f = x
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = torch.cat([x, x_flatten], dim = 1)
        x = self.fc(x)
        self.boundary_maps = self._get_boundary_location()

        b = self.boundary_forward()

        b_flatten = self.last_pool(b)
        b_flatten = b_flatten.view(b_flatten.size(0), -1)

        b_f = b
        b = self.avgpool(b)
        b = b.view(b.size(0), -1)
        b = torch.cat([b, b_flatten], dim = 1)
        b = self.boundary_fc(b)

        ensemble = torch.cat([x_f, b_f], dim = 1)
        ensemble = self.ensemble_relu(ensemble)

        ensemble_flatten = self.last_pool(ensemble)
        ensemble_flatten = ensemble_flatten.view(ensemble_flatten.size(0), -1)

        ensemble = self.avgpool(ensemble)
        ensemble = ensemble.view(ensemble.size(0), -1)
        ensemble = torch.cat([ensemble, ensemble_flatten], dim = 1)
        ensemble = self.ensemble_fc(ensemble)

        return x, b, ensemble


# if __name__ == '__main__' :

#     device = torch.device(0)
#     model = ResNet(BasicBlock, [2, 2, 2, 2], [128, 256, 512], device).to(device)
#     print(summary(model, (3, 224, 224), device = 'cuda'))