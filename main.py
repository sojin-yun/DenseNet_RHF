import sys
import os

import copy
import torchsummary
from utils.select_model import Select_Model
from utils.random_seed import Fix_Randomness
from utils.parse_args import Parsing_Args
from utils.load_data import CustomDataLoader
from utils.corrupt import EvaluateMCE
from run.train import TrainingEnsemble, TrainingBaseline
from run.eval import Evaluation
from run.heatmap import RunGradCAM
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
    n_device = 'cuda:{}'.format(flags['device']) if flags['device'] != 'cpu' else 'cpu'
    device = torch.device(n_device)
    abs_path = '/home/NAS_mount/sjlee/RHF' if flags['server'] else '.'

    # Load Dataset
    if flags['mode'] != 'imagenet_c' :
        data_loader = CustomDataLoader(flags)()

        # Model Selection
        if not flags['baseline'] :
            model = Select_Model(args = flags, device = device).ensemble_model(model = flags['model'])
        else :
            model = Select_Model(args = flags, device = device).baseline_model(model = flags['model'])

        # Use Pretrained Weights
        if flags['pretrained'] :
            if flags['baseline'] :
                params = torch.load('{0}/weights/baseline/{1}_{2}.pth'.format(abs_path, flags['model'], flags['data']), map_location = device)
            else :
                params = torch.load('{0}/weights/ensemble/{1}_{2}.pth'.format(abs_path, flags['model'], flags['data']), map_location = device)

            model_params = model.state_dict()
            model_params.update(params)
            model.load_state_dict(model_params)
            print('Pretrained weights are loaded.')


    # Select what mode you want to run : ['train', 'eval', 'cam']
    if flags['mode'] == 'train' :
        if not flags['baseline'] :
            TrainingEnsemble(flags, model ,data_loader, device)()
        else :
            TrainingBaseline(flags, model, data_loader, device)()

    elif flags['mode'] == 'eval' :
        load_checkpoint = flags['weight']
        if load_checkpoint != None :
            checkpoint = torch.load('{0}/weights/evaluation/{1}'.format(abs_path, flags['weight']), map_location = device)
            params = checkpoint['state_dict']
            model_params = model.state_dict()
            model_params.update(params)
            model.load_state_dict(model_params)
            print('Pretrained weights are loaded for evaluation.')
        Evaluation(flags, model ,data_loader, device)()

    elif flags['mode'] == 'cam' :
        cam = RunGradCAM(flags, data_loader)
        cam.run()

    elif flags['mode'] == 'imagenet_c' :
        baseline_model = Select_Model(args = flags, device = device).baseline_model(model = flags['model'])      
        ensemble_model = Select_Model(args = flags, device = device).ensemble_model(model = flags['model'])
        eval_mce = EvaluateMCE(flags, baseline_model, ensemble_model, device)
        eval_mce.run()


if __name__ == '__main__' :
    drive(sys.argv[1:])