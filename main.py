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

        # model_params = model1.state_dict()
        # model_params.update(params)
        # model1.load_state_dict(model_params)

        pretrained_param, dict_key_conv, dict_key_bn, dict_key_linear = download_params()
        model = put_parameter(model, pretrained_param, dict_key_conv, dict_key_bn, dict_key_linear, True)

        #model.state_dict = copy.deepcopy(model1.state_dict())

        # for i, j in zip(list(model1.parameters()), list(model.parameters())) :
        #     if (i == j).all() :
        #         print(i.shape, j.shape)
        #     else :
        #         print('different')
        # return


    # Select what mode you want to run
    if flags['mode'] == 'train' :
        if not flags['baseline'] :
            TrainingEnsemble(flags, model ,data_loader, device)()
        else :
            TrainingBaseline(flags, model, data_loader, device)()
    elif flags['mode'] == 'eval' :
        Evaluation(flags, model ,data_loader, device)()


def download_params():

    state_dict = dict(torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'))

    dict_key_conv = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7', 'conv8', 'conv9', 'conv10', 'conv11', 'conv12', 'conv13']
    dict_key_bn = ['bn1', 'bn2', 'bn3', 'bn4', 'bn5', 'bn6', 'bn7', 'bn8', 'bn9', 'bn10', 'bn11', 'bn12', 'bn13']
    dict_key_linear = ['fc1', 'fc2', 'fc3']
    state_dict_keys = state_dict.keys()

    pretrained_param = {}
    pretrained_param['conv'] = {}
    pretrained_param['bn'] = {}
    pretrained_param['linear'] = {}

    for idx, k in enumerate(state_dict_keys):

        dict_idx = idx // 6
        weight_idx = idx % 6
        fc_dict_idx = (idx-78) // 2
        fc_weight_idx = idx % 2
        
        if idx < 78 :
                
            if weight_idx == 0 :
                pretrained_param['conv'][dict_key_conv[dict_idx]] = {}
                pretrained_param['conv'][dict_key_conv[dict_idx]]['weight'] = state_dict[k]
            elif weight_idx == 1 :
                pretrained_param['conv'][dict_key_conv[dict_idx]]['bias'] = state_dict[k]
            elif weight_idx == 2 :
                pretrained_param['bn'][dict_key_bn[dict_idx]] = {}
                pretrained_param['bn'][dict_key_bn[dict_idx]]['weight'] = state_dict[k]
            elif weight_idx == 3 :
                pretrained_param['bn'][dict_key_bn[dict_idx]]['bias'] = state_dict[k]
            elif weight_idx == 4 :
                pretrained_param['bn'][dict_key_bn[dict_idx]]['running_mean'] = state_dict[k]
            elif weight_idx == 5 :
                pretrained_param['bn'][dict_key_bn[dict_idx]]['running_var'] = state_dict[k]
            else :
                print('Error is occured!')
                return

        else :
            if fc_weight_idx == 0 :
                pretrained_param['linear'][dict_key_linear[fc_dict_idx]] = {}
                pretrained_param['linear'][dict_key_linear[fc_dict_idx]]['weight'] = state_dict[k]
            elif fc_weight_idx == 1 :
                pretrained_param['linear'][dict_key_linear[fc_dict_idx]]['bias'] = state_dict[k]
            else :
                print('Error is occured!')
                return
    
    return pretrained_param, dict_key_conv, dict_key_bn, dict_key_linear


def put_parameter(model, param_dict, dict_key_conv, dict_key_bn, dict_key_linear, is_subset):

    conv_idx, bn_idx, fc_idx = 0, 0, 0

    for m in model.features.modules():

        if isinstance(m, nn.Conv2d) :
            with torch.no_grad():
                m.weight.data = nn.Parameter(param_dict['conv'][dict_key_conv[conv_idx]]['weight'])
                m.bias.data = nn.Parameter(param_dict['conv'][dict_key_conv[conv_idx]]['bias'])
                print(dict_key_conv[conv_idx], 'is setted.')
                conv_idx += 1
        
        elif isinstance(m, nn.BatchNorm2d) :
            with torch.no_grad():
                m.weight.data = nn.Parameter(param_dict['bn'][dict_key_bn[bn_idx]]['weight'])
                m.bias.data = nn.Parameter(param_dict['bn'][dict_key_bn[bn_idx]]['bias'])
                m.running_mean.data = param_dict['bn'][dict_key_bn[bn_idx]]['running_mean']
                m.running_var.data = param_dict['bn'][dict_key_bn[bn_idx]]['running_var']
                print(dict_key_bn[bn_idx], 'is setted.')
                bn_idx += 1

    if not is_subset :

        for m in model.classifier.modules():
            if isinstance(m, nn.Linear) :
                with torch.no_grad():
                    m.weight = nn.Parameter(param_dict['linear'][dict_key_linear[fc_idx]]['weight'])
                    m.bias = nn.Parameter(param_dict['linear'][dict_key_linear[fc_idx]]['bias'])
                    print(dict_key_linear[fc_idx], 'is setted.')
                    fc_idx += 1
        print('Load fully-connected-weight and bias')

    return model

if __name__ == '__main__' :
    drive(sys.argv[1:])