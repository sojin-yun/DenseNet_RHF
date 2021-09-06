
from model.ensemble.resnet import ResNet_ensemble, BasicBlock_ensemble
from model.ensemble.resnet50 import ResNet_ensemble_50, BasicBlock_ensemble_50

class Select_Model :
    def __init__(self, args, device) :

        self.model = args['model']
        self.is_baseline = args['baseline']
        self.device = device

    def __call__(self) :
        if self.is_baseline :
            self.baseline_models(self.model)

    def baseline_models(self, model) :
        
        if model == 'resnet18' :
            return ResNet_ensemble(BasicBlock_ensemble, [2, 2, 2, 2], [128, 256, 512], 100, None, self.device)

        elif model == 'resnet50' :
            return ResNet_ensemble_50(BasicBlock_ensemble_50, [3, 4, 6, 4], [128, 256, 512], 100, None, self.device)

        elif model == 'resnet101' :
            pass

        elif model == 'vgg16' :
            pass

        elif model == 'vgg19' :
            pass

        elif model == 'densenet121' :
            pass

        elif model == 'densenet169' :
            pass
