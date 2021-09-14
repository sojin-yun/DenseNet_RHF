
from model.ensemble.resnet import ResNet_ensemble, BasicBlock_ensemble
from model.ensemble.resnet_deeper import ResNet_ensemble_deeper, BasicBlock_ensemble_deeper
from model.ensemble.densenet import DenseNet_ensemble, Bottleneck_ensemble
from model.ensemble.vgg import VGG_ensemble
from model.baseline.resnet import ResNet, BasicBlock
from model.baseline.resnet_deeper import ResNet_deeper, BasicBlock_deeper
from model.baseline.densenet import DenseNet, Bottleneck
from model.baseline.vgg import VGG

class Select_Model :
    def __init__(self, args, device = 'cpu') :

        self.model = args['model']
        self.is_baseline = args['baseline']
        self.data = args['data']
        self.device = device
        if self.data == 'cifar100' : self.low_resolution = True
        else : self.low_resolution = False

    def __call__(self) :
        if self.is_baseline :
            self.baseline_models(self.model)


    def ensemble_model(self, model) :
        
        if model == 'resnet18' :
            return ResNet_ensemble(BasicBlock_ensemble, [2, 2, 2, 2], [128, 256, 512], 100, None, self.device)

        elif model == 'resnet50' :
            return ResNet_ensemble_deeper(BasicBlock_ensemble_deeper, [3, 4, 6, 3], [128, 256, 512], 100, None, self.device, low_resolution = self.low_resolution)

        elif model == 'resnet101' :
            return ResNet_ensemble_deeper(BasicBlock_ensemble_deeper, [3, 4, 23, 3], [128, 256, 512], 100, None, self.device, low_resolution = self.low_resolution)

        elif model == 'resnet152' :
            return ResNet_ensemble_deeper(BasicBlock_ensemble_deeper, [3, 8, 36, 3], [128, 256, 512], 100, None, self.device, low_resolution = self.low_resolution)

        elif model == 'vgg16' :
            return VGG_ensemble(model_key = '16', num_classes = 100, device = self.device, data = self.data)

        elif model == 'vgg19' :
            return VGG_ensemble(model_key = '19', num_classes = 100, device = self.device, data = self.data)

        elif model == 'densenet121' :
            return DenseNet_ensemble(Bottleneck_ensemble, [6, 12, 24, 16], [128, 256, 512], 32, device = self.device, low_resolution = self.low_resolution)

        elif model == 'densenet169' :
            return DenseNet_ensemble(Bottleneck_ensemble, [6, 12, 32, 32], [128, 256, 512], 32, device = self.device, low_resolution = self.low_resolution)

        elif model == 'densenet201' :
            return DenseNet_ensemble(Bottleneck_ensemble, [6, 12, 48, 32], [128, 256, 512], 32, device = self.device, low_resolution = self.low_resolution)


    def baseline_model(self, model) :
        
        if model == 'resnet18' :
            return ResNet(BasicBlock, [2, 2, 2, 2], 100, None)

        elif model == 'resnet50' :
            return ResNet_deeper(BasicBlock_deeper, [3, 4, 6, 3], 100, None, low_resolution = self.low_resolution)

        elif model == 'resnet101' :
            return ResNet_deeper(BasicBlock_deeper, [3, 4, 23, 3], 100, None, low_resolution = self.low_resolution)

        elif model == 'resnet152' :
            return ResNet_deeper(BasicBlock_deeper, [3, 8, 36, 3], 100, None, low_resolution = self.low_resolution)

        elif model == 'vgg16' :
            return VGG(model_key = '16', num_classes=100, data = self.data)

        elif model == 'vgg19' :
            return VGG(model_key = '19', num_classes=100, data = self.data)

        elif model == 'densenet121' :
            return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=32, low_resolution = self.low_resolution)

        elif model == 'densenet169' :
            return DenseNet(Bottleneck, [6, 12, 32, 32], growth_rate=32, low_resolution = self.low_resolution)

        elif model == 'densenet201' :
            return DenseNet(Bottleneck, [6, 12, 48, 32], growth_rate=32, low_resolution = self.low_resolution)