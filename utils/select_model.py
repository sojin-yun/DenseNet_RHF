
from model.ensemble.resnet import ResNet_ensemble, BasicBlock_ensemble
from model.ensemble.resnet_deeper import ResNet_ensemble_deeper, BasicBlock_ensemble_deeper
from model.ensemble.densenet import DenseNet_ensemble, Bottleneck_ensemble
from model.ensemble.vgg import VGG_ensemble
from model.ensemble.vgg_recursive import VGG_ensemble_recursive
#from model.ensemble.squeezenet import SqueezeNet, Fire, _squeezenet
from model.baseline.resnet import ResNet, BasicBlock
from model.baseline.resnet_deeper import ResNet_deeper, BasicBlock_deeper
from model.baseline.densenet import DenseNet, Bottleneck
from model.baseline.vgg import VGG
from model.baseline.alexnet import AlexNet
from model.baseline.squeezenet import SqueezeNet, Fire, _squeezenet

class Select_Model :
    def __init__(self, args, device = 'cpu') :

        self.model = args['model']
        self.is_baseline = args['baseline']
        self.data = args['data']
        self.device = device
        if self.data == 'cifar100' or self.data == 'mnist' : self.low_resolution = True
        else : self.low_resolution = False
        if self.data == 'kidney_stone' : self.numclasses = 2
        elif self.data == 'mnist' : self.numclasses = 10
        else : self.numclasses = 100

    def __call__(self) :
        if self.is_baseline :
            self.baseline_models(self.model)


    def ensemble_model(self, model) :
        
        if model == 'resnet18' :
            return ResNet_ensemble(BasicBlock_ensemble, [2, 2, 2, 2], [128, 256, 512], self.numclasses, None, self.device)

        elif model == 'resnet50' :
            return ResNet_ensemble_deeper(BasicBlock_ensemble_deeper, [3, 4, 6, 3], [128, 256, 512], self.numclasses, None, self.device, low_resolution = self.low_resolution)

        elif model == 'resnet101' :
            return ResNet_ensemble_deeper(BasicBlock_ensemble_deeper, [3, 4, 23, 3], [128, 256, 512], self.numclasses, None, self.device, low_resolution = self.low_resolution)

        elif model == 'resnet152' :
            return ResNet_ensemble_deeper(BasicBlock_ensemble_deeper, [3, 8, 36, 3], [128, 256, 512], self.numclasses, None, self.device, low_resolution = self.low_resolution)

        elif model == 'vgg16' :
            return VGG_ensemble(model_key = '16', num_classes = self.numclasses, device = self.device, data = self.data)

        elif model == 'vgg19' :
            return VGG_ensemble(model_key = '19', num_classes = self.numclasses, device = self.device, data = self.data)

        elif model == 'vgg16_recursive' :
            return VGG_ensemble_recursive(model_key = '16', num_classes = self.numclasses, device = self.device, data = self.data)

        elif model == 'densenet121' :
            return DenseNet_ensemble(Bottleneck_ensemble, [6, 12, 24, 16], [128, 256, 512], 32, num_class = self.numclasses, device = self.device, low_resolution = self.low_resolution)

        elif model == 'densenet169' :
            return DenseNet_ensemble(Bottleneck_ensemble, [6, 12, 32, 32], [128, 256, 640], 32, num_class = self.numclasses, device = self.device, low_resolution = self.low_resolution)

        elif model == 'densenet201' :
            return DenseNet_ensemble(Bottleneck_ensemble, [6, 12, 48, 32], [128, 256, 896], 32, num_class = self.numclasses, device = self.device, low_resolution = self.low_resolution)


    def baseline_model(self, model) :
        
        if model == 'resnet18' :
            return ResNet(BasicBlock, [2, 2, 2, 2], self.numclasses, None)

        elif model == 'resnet50' :
            return ResNet_deeper(BasicBlock_deeper, [3, 4, 6, 3], self.numclasses, None, low_resolution = self.low_resolution)

        elif model == 'resnet101' :
            return ResNet_deeper(BasicBlock_deeper, [3, 4, 23, 3], self.numclasses, None, low_resolution = self.low_resolution)

        elif model == 'resnet152' :
            return ResNet_deeper(BasicBlock_deeper, [3, 8, 36, 3], self.numclasses, None, low_resolution = self.low_resolution)

        elif model == 'vgg16' :
            return VGG(model_key = '16', num_classes=self.numclasses, data = self.data)

        elif model == 'vgg19' :
            return VGG(model_key = '19', num_classes=self.numclasses, data = self.data)

        elif model == 'densenet121' :
            return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=32, num_class = self.numclasses, low_resolution = self.low_resolution)

        elif model == 'densenet169' :
            return DenseNet(Bottleneck, [6, 12, 32, 32], growth_rate=32, num_class = self.numclasses, low_resolution = self.low_resolution)

        elif model == 'densenet201' :
            return DenseNet(Bottleneck, [6, 12, 48, 32], growth_rate=32, num_class = self.numclasses, low_resolution = self.low_resolution)

        elif model == 'alexnet' :
            return AlexNet(num_classes=100)

        elif model == 'squeezenet10' :
            return _squeezenet('1_0')

        elif model == 'squeezenet11' :
            return _squeezenet('1_1')
