import sys
import os

from torch.utils.tensorboard import SummaryWriter
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
    save_path = flags['dst']
    save_file = flags['file']

    model_name, data = flags['model'], flags['data']
    params = torch.load('{0}/weights/baseline/{1}_{2}.pth'.format(abs_path, model_name, data), map_location = device)

    default_path = '/home/NAS_mount/sjlee/RHF/Save_parameters/lung/' if flags['server'] else './Save_parameters/lung/'
    if not os.path.isdir(os.path.join(default_path, save_path)) :
            os.mkdir(os.path.join(default_path, save_path))

    folder = 'tensor_board'
    os.mkdir(os.path.join(default_path, save_path, folder))
    ts_board = SummaryWriter(log_dir = os.path.join(default_path, save_path, folder))
    

    # Model Selection
    if flags['model'] == 'resnet50' :
        cam_model = ResNet_deeper(BasicBlock_deeper, [3, 4, 6, 3], 50, None, low_resolution = False)
        # model_params = cam_model.state_dict()
        # model_params.update(params)
        # cam_model.load_state_dict(model_params)
        # print('Pretrained weights are loaded.')
        gain_model = GAIN(device, cam_model, 152)


    elif flags['model'] == 'densenet121' :
        cam_model = DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=32, num_class = 50, low_resolution = False)
        # model_params = cam_model.state_dict()
        # model_params.update(params)
        # cam_model.load_state_dict(model_params)
        # print('Pretrained weights are loaded.')
        gain_model = GAIN(device, cam_model, 492)


    cam_model, gain_model = cam_model.to(device), gain_model.to(device)
    #cam_model = cam_model.to(device)

    # Use Pretrained Weights
    print('Training Start')

    best_valid_score = 0.
    best_epoch =  0

    for i in range(epoch) :

            train_loss, valid_loss = 0.0, 0.0
            train_acc, valid_acc = 0.0, 0.0

            print('------------[Epoch:{}]-------------'.format(i+1))
            gain_model.train()
            #cam_model.train()

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
                
                train_acc += (torch.sum(pred == train_target.data).item()*(100.0 / batch_size))
                ts_board.add_scalar('train/total_loss', total_loss.item(), i * n_train_batchs + train_iter)
                ts_board.add_scalar('train/cl_loss', loss_cl.item(), i * n_train_batchs + train_iter)
                ts_board.add_scalar('train/am_loss', loss_am.item(), i * n_train_batchs + train_iter)

                # cam_model.optimizer.zero_grad()

                # train_output = cam_model(train_data)

                # t_loss = cam_model.loss(train_output, train_target)

                # t_loss.backward()

                # cam_model.optimizer.step()
                
                # _, pred = torch.max(train_output, dim = 1)
            
                # train_loss += t_loss.item()
                # train_acc += (torch.sum(pred == train_target.data).item()*(100.0 / batch_size))

                # ts_board.add_scalar('Loss/train', t_loss.item(), i * n_train_batchs + train_iter)


            #with torch.enable_grad() :
            with torch.no_grad() :

                gain_model.eval()
                #cam_model.eval()

                for valid_iter, (v_samples) in enumerate(tqdm(valid_loader, desc="{:17s}".format('Evaluation State'), mininterval=0.01)) :

                    if flags['mask'] : valid_data, valid_target, _ = v_samples
                    else : valid_data, valid_target = v_samples

                    if device != None : valid_data, valid_target = valid_data.to(device), valid_target.to(device)

                    gain_model.model.optimizer.zero_grad()

                    v_total_loss, v_loss_cl, v_loss_am, Ac, v_preds = gain_model(valid_data, valid_target)

                    valid_acc += (torch.sum(v_preds == valid_target.data).item()*(100.0 / batch_size))
                    ts_board.add_scalar('valid/valid_total_loss', v_total_loss.item(), i * n_valid_batchs + valid_iter)
                    ts_board.add_scalar('valid/valid_cl_loss', v_loss_cl.item(), i * n_valid_batchs + valid_iter)
                    ts_board.add_scalar('valid/valid_am_loss', v_loss_am.item(), i * n_valid_batchs + valid_iter)

                    # cam_model.optimizer.zero_grad()

                    # valid_output = cam_model(valid_data)

                    # v_loss = cam_model.loss(valid_output, valid_target)

                    # _, v_pred = torch.max(valid_output, dim = 1)

                    # valid_acc += (torch.sum(v_pred == valid_target.data)).item()*(100.0 / batch_size)

                    # ts_board.add_scalar('Loss/valid', v_loss.item(), i * n_valid_batchs + valid_iter)


            avg_train_acc = train_acc/n_train_batchs
            avg_valid_acc = valid_acc/n_valid_batchs

            if avg_valid_acc > best_valid_score : 
                best_valid_score = avg_valid_acc
                best_epoch = i

            ts_board.add_scalars('Accuracy', {'train_acc' : avg_train_acc, 
                                              'valid_acc' : avg_valid_acc}, i)

            curr_lr = gain_model.model.optimizer.param_groups[0]['lr']
            #curr_lr = cam_model.optimizer.param_groups[0]['lr']
            #training_result = 'epoch.{0:3d} \t train_ac : {1:.4f}% \t  valid_ac : {2:.4f}% \t lr : {3:.6f}\n'.format(i+1, avg_train_acc, avg_valid_acc, curr_lr)

            gain_model.model.scheduler.step()
            #cam_model.scheduler.step()

            epoch_model_params = {
                'epoch' : i+1,
                'state_dict' : gain_model.state_dict()
                #'state_dict' : cam_model.state_dict()
            }
            torch.save(epoch_model_params, os.path.join(default_path, save_path, save_file+'_{}_epoch.pt'.format(i+1)))
            print('epoch.{0:3d} \t train_ac : {1:.4f}% \t  valid_ac : {2:.4f}% \t lr : {3:.6f}\n'.format(i+1, avg_train_acc, avg_valid_acc, curr_lr))
            #print('epoch.{0:3d} \t train_ac : {1:.4f}% \t  valid_ac : {2:.4f}% \t lr : {3:.6f}\n'.format(i+1, avg_train_acc, avg_valid_acc, curr_lr))

if __name__ == '__main__' :
    drive(sys.argv[1:])