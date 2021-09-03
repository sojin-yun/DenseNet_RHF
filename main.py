import sys
import os
from utils.random_seed import Fix_Randomness
from utils.parse_args import Parsing_Args
from utils.load_data import CustomDataLoader
from run.train import TrainingEnsemble
from model.resnet import ResNet_ensemble, BasicBlock_ensemble
from torchsummary import summary
import torch

def drive(args) :

    flags = Parsing_Args(args)

    device = torch.device(flags['device'])
    data_loader = CustomDataLoader(flags)()
    model = ResNet_ensemble(block = BasicBlock_ensemble, layers = [2, 2, 2, 2], boundary_layers = [128, 256, 512], num_classes = 100, device = device)
    TrainingEnsemble(flags, model ,data_loader, device)()

if __name__ == '__main__' :
    drive(sys.argv)