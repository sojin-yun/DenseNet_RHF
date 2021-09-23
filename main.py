import sys
import os

import copy
import torchsummary
from utils.select_model import Select_Model
from utils.random_seed import Fix_Randomness
from utils.parse_args import Parsing_Args
from utils.load_data import CustomDataLoader
from run.train import TrainingEnsemble, TrainingBaseline
from run.eval import Evaluation
from model.ensemble.resnet import ResNet_ensemble, BasicBlock_ensemble
from model.baseline.resnet import ResNet, BasicBlock
from model.ensemble.resnet_deeper import BasicBlock_ensemble_deeper, ResNet_ensemble_deeper
from model.baseline.resnet_deeper import BasicBlock_deeper, ResNet_deeper
from torchsummary import summary
import torch
import torch.nn as nn

def drive(args) :

    Fix_Randomness(42)

    flags = Parsing_Args(args)

    device = torch.device(flags['device'])
    data_loader = CustomDataLoader(flags)()

    # Model Selection
    if not flags['baseline'] :
        model = Select_Model(args = flags, device = device).ensemble_model(model = flags['model'])      
    else :
        model = Select_Model(args = flags, device = device).baseline_model(model = flags['model'])      

    # Use Pretrained Weights
    if flags['pretrained'] :
        if flags['baseline'] :
            params = torch.load('./weights/baseline/{0}_{1}.pth'.format(flags['model'], flags['data']), map_location = device)
        else :
            params = torch.load('./weights/ensemble/{0}_{1}.pth'.format(flags['model'], flags['data']), map_location = device)

        model_params = model.state_dict()
        model_params.update(params)
        model.load_state_dict(model_params)
        print('Pretrained weights are loaded.')


    # Select what mode you want to run
    if flags['mode'] == 'train' :
        if not flags['baseline'] :
            TrainingEnsemble(flags, model ,data_loader, device)()
        else :
            TrainingBaseline(flags, model, data_loader, device)()
    elif flags['mode'] == 'eval' :
        load_checkpoint = flags['weight']
        if load_checkpoint != None :
            checkpoint = torch.load('./weights/evaluation/{0}'.format(flags['weight']), map_location = device)
            params = checkpoint['state_dict']
            model_params = model.state_dict()
            model_params.update(params)
            model.load_state_dict(model_params)
            print('Pretrained weights are loaded for evaluation.')
        Evaluation(flags, model ,data_loader, device)()


if __name__ == '__main__' :
    drive(sys.argv[1:])