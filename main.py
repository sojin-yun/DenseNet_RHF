import sys
import os
from utils.random_seed import Fix_Randomness
from utils.parse_args import Parsing_Args
from utils.load_data import CustomDataLoader
from run.train import TrainingEnsemble, TrainingBaseline
from model.ensemble.resnet import ResNet_ensemble, BasicBlock_ensemble
from model.baseline.resnet import ResNet, BasicBlock
from model.ensemble.resnet_deeper import BasicBlock_ensemble_deeper, ResNet_ensemble_deeper
from model.baseline.resnet_deeper import BasicBlock_deeper, ResNet_deeper
from torchsummary import summary
import torch

def drive(args) :

    flags = Parsing_Args(args)

    device = torch.device(flags['device'])
    data_loader = CustomDataLoader(flags)()

    if not flags['baseline'] :
        model = ResNet_ensemble_deeper(block = BasicBlock_ensemble_deeper, layers = [3, 4, 6, 3], boundary_layers = [128, 256, 512], num_classes = 100, device = device, low_resolution = False)
    else :
        model = ResNet_deeper(block = BasicBlock_deeper, layers = [3, 4, 6, 3], num_classes = 100, low_resolution = False)

    if not flags['baseline'] :
        TrainingEnsemble(flags, model ,data_loader, device)()
    else :
        TrainingBaseline(flags, model, data_loader, device)()

if __name__ == '__main__' :
    drive(sys.argv)