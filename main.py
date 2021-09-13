import sys
import os

import torchsummary
from utils.select_model import Select_Model
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

    Fix_Randomness(42)

    flags = Parsing_Args(args)

    device = torch.device(flags['device'])
    data_loader = CustomDataLoader(flags)()

    if not flags['baseline'] :
        model = Select_Model(args = flags, device = device).ensemble_model(model = flags['model'])
    else :
        model = Select_Model(args = flags, device = device).baseline_model(model = flags['model'])

    if not flags['baseline'] :
        TrainingEnsemble(flags, model ,data_loader, device)()
    else :
        TrainingBaseline(flags, model, data_loader, device)()


if __name__ == '__main__' :
    drive(sys.argv)