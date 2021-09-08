
from model.ensemble.resnet import ResNet_ensemble, BasicBlock_ensemble
from model.ensemble.resnet_deeper import ResNet_ensemble_deeper, BasicBlock_ensemble_deeper
from model.baseline.resnet import ResNet, BasicBlock
from model.baseline.resnet_deeper import ResNet_deeper, BasicBlock_deeper

class Select_Model :
    def __init__(self, args, device) :

        self.model = args['model']
        self.is_baseline = args['baseline']
        self.device = device

    def __call__(self) :
        if self.is_baseline :
            self.baseline_models(self.model)

    def ensemble_model(self, model) :
        
        if model == 'resnet18' :
            return ResNet_ensemble(BasicBlock_ensemble, [2, 2, 2, 2], [128, 256, 512], 100, None, self.device)

        elif model == 'resnet50' :
            return ResNet_ensemble_deeper(BasicBlock_ensemble_deeper, [3, 4, 6, 3], [128, 256, 512], 100, None, self.device)

        elif model == 'resnet101' :
            return ResNet_ensemble_deeper(BasicBlock_ensemble_deeper, [3, 4, 23, 3], [128, 256, 512], 100, None, self.device)

        elif model == 'vgg16' :
            pass

        elif model == 'vgg19' :
            pass

        elif model == 'densenet121' :
            pass

        elif model == 'densenet169' :
            pass

    def baseline_model(self, model) :
        
        if model == 'resnet18' :
            return ResNet(BasicBlock, [2, 2, 2, 2], 100, None)

        elif model == 'resnet50' :
            return ResNet_deeper(BasicBlock_deeper, [3, 4, 6, 3], 100, None)

        elif model == 'resnet101' :
            return ResNet_deeper(BasicBlock_deeper, [3, 4, 23, 3], 100, None)

        elif model == 'vgg16' :
            pass

        elif model == 'vgg19' :
            pass

        elif model == 'densenet121' :
            pass

        elif model == 'densenet169' :
            pass
