import sys
import os

from utils.select_model import Select_Model
from utils.random_seed import Fix_Randomness
from utils.parse_args import Parsing_Args
from utils.load_data import CustomDataLoader
from model.baseline.resnet_deeper import BasicBlock_deeper, ResNet_deeper
from model.baseline.densenet import DenseNet, Transition, Bottleneck
from GAIN import GAIN
import torch
import torch.nn as nn
from tqdm import tqdm

def drive(args) :

    Fix_Randomness(42)

    flags = Parsing_Args(args)
    n_device = 'cuda:{}'.format(flags['device']) if flags['device'] != 'cpu' else 'cpu'
    device = torch.device(n_device)
    epoch = flags['epoch']
    data_loader = CustomDataLoader(flags)()
    train_loader, valid_loader = data_loader

    abs_path = '/home/NAS_mount/sjlee/RHF' if flags['server'] else '.'

    model_name, data = flags['model'], flags['data']
    params = torch.load('{0}/weights/baseline/{1}_{2}.pth'.format(abs_path, model_name, data), map_location = device)


    # Model Selection
    if flags['model'] == 'resnet50' :
        cam_model = ResNet_deeper(BasicBlock_deeper, [3, 4, 6, 3], 50, None, low_resolution = False)
        model_params = cam_model.state_dict()
        model_params.update(params)
        cam_model.load_state_dict(model_params)
        print('Pretrained weights are loaded.')
        gain_model = GAIN(device, cam_model, 152)

    elif flags['model'] == 'densenet121' :
        cam_model = DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=32, num_class = 50, low_resolution = False)
        model_params = cam_model.state_dict()
        model_params.update(params)
        cam_model.load_state_dict(model_params)
        print('Pretrained weights are loaded.')
        gain_model = GAIN(device, cam_model, 492)


    cam_model, gain_model = cam_model.to(device), gain_model.to(device)

    # Use Pretrained Weights
    print('Training Start')
    for i in range(epoch) :

            train_loss, valid_loss = 0.0, 0.0
            train_acc, valid_acc = 0.0, 0.0

            print('------------[Epoch:{}]-------------'.format(i+1))
            gain_model.train()

            n_train_batchs = len(train_loader)
            n_valid_batchs = len(valid_loader)
            batch_size = flags['batch_size']

            for train_iter, (samples) in enumerate(tqdm(train_loader, desc="{:17s}".format('Training State'), mininterval=0.01)) :

                
                if flags['mask'] : train_data, train_target, _ = samples
                else : train_data, train_target = samples

                if device != None : train_data, train_target = train_data.to(device), train_target.to(device)
                
                total_loss, loss_cl, loss_am, _, pred = gain_model(train_data, train_target)

                total_loss.backward()

                gain_model.model.optimizer.step()
                
                train_acc = (torch.sum(pred == train_target.data).item()*(100.0 / batch_size))
                print('loss_total : {:.4f} \t loss_cl : {:.4f} \t loss_am : {:.4f} \t accuracy : {:.4f}%'.format(total_loss, loss_cl, loss_am, train_acc))

            with torch.enable_grad() :

                gain_model.eval()

                for valid_iter, (v_samples) in enumerate(tqdm(valid_loader, desc="{:17s}".format('Evaluation State'), mininterval=0.01)) :

                    if flags['mask'] : valid_data, valid_target, _ = v_samples
                    else : valid_data, valid_target = v_samples

                    if device != None : valid_data, valid_target = valid_data.to(device), valid_target.to(device)

                    gain_model.model.optimizer.zero_grad()

                    v_total_loss, v_loss_cl, v_loss_am, Ac, preds = gain_model(valid_data, valid_target)

            training_result = 'epoch.{0:3d} \t train_ac : {1:.4f}% \t  valid_ac : {2:.4f}% \t lr : {3:.6f}\n'.format(i+1, avg_train_acc, avg_valid_acc, curr_lr)

            avg_train_acc = train_acc/n_train_batchs
            avg_valid_acc = valid_acc/n_valid_batchs
            avg_tarin_loss = train_loss/n_train_batchs
            avg_valid_loss = valid_loss/n_valid_batchs

            curr_lr = gain_model.model.optimizer.param_groups[0]['lr']

            gain_model.model.scheduler.step()

if __name__ == '__main__' :
    drive(sys.argv[1:])