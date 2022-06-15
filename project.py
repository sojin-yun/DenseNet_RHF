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
from utils.image_process import InverseNormalize
from utils.grad_cam import GradCAM
import numpy as np
import matplotlib.pyplot as plt
import cv2

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
    #params = torch.load('{0}/weights/baseline/{1}_{2}.pth'.format(abs_path, model_name, data), map_location = device)
    cam_params = torch.load('{0}/weights/evaluation/densenet121_cam_lung.pt'.format(abs_path), map_location = device)['state_dict']
    gain_params = torch.load('{0}/weights/evaluation/densenet121_gain_lung.pt'.format(abs_path), map_location = device)['state_dict']

    default_path = '/home/NAS_mount/sjlee/RHF/Save_parameters/lung/' if flags['server'] else './Save_parameters/lung/'
    if not os.path.isdir(os.path.join(default_path, save_path)) :
            os.mkdir(os.path.join(default_path, save_path))

    folder = 'tensor_board'
    os.mkdir(os.path.join(default_path, save_path, folder))
    ts_board = SummaryWriter(log_dir = os.path.join(default_path, save_path, folder))
    

    # Model Selection
    if flags['model'] == 'resnet50' :
        # cam_model = ResNet_deeper(BasicBlock_deeper, [3, 4, 6, 3], 2, None, low_resolution = False)
        # # model_params = cam_model.state_dict()
        # # model_params.update(params)
        # # cam_model.load_state_dict(model_params)
        # # print('Pretrained weights are loaded.')
        # gain_model = GAIN(device, cam_model, 152)
        cam_model = ResNet_deeper(BasicBlock_deeper, [3, 4, 6, 3], 100, None, low_resolution = False)
        gain_backbone = ResNet_deeper(BasicBlock_deeper, [3, 4, 6, 3], 2, None, low_resolution = False)
        model_params = cam_model.state_dict()
        model_params.update(cam_params)
        cam_model.load_state_dict(model_params)
        
        gain_model = GAIN(device, gain_backbone, 152)
        model_params = gain_model.state_dict()
        model_params.update(gain_params)
        gain_model.load_state_dict(model_params)
        print('Pretrained weights are loaded.')


    elif flags['model'] == 'densenet121' :
        # cam_model = DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=32, num_class = 2, low_resolution = False)
        # # model_params = cam_model.state_dict()
        # # model_params.update(params)
        # # cam_model.load_state_dict(model_params)
        # # print('Pretrained weights are loaded.')
        # gain_model = GAIN(device, cam_model, 492)
        cam_model = DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=32, num_class = 100, low_resolution = False)
        gain_backbone = DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=32, num_class = 2, low_resolution = False)
        model_params = cam_model.state_dict()
        model_params.update(cam_params)
        cam_model.load_state_dict(model_params)
        
        gain_model = GAIN(device, gain_backbone, 492)
        model_params = gain_model.state_dict()
        model_params.update(gain_params)
        gain_model.load_state_dict(model_params)
        print('Pretrained weights are loaded.')


    # cam_model, gain_model = cam_model.to(device), gain_model.to(device)
    # #cam_model = cam_model.to(device)

    grad_cam = GradCAM(model = cam_model, hooked_layer = 152, device = device, ensemble = False)
    grad_gain = GradCAM(model = gain_model.model, hooked_layer = 152, device = device, ensemble = False)
    upsample = nn.Upsample(size = 512, mode = 'bilinear', align_corners = False)
    #inverse_norm = InverseNormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    inverse_norm = InverseNormalize((0.27, 0.27, 0.27), (0.309, 0.309, 0.309))

    # Use Pretrained Weights
    print('Training Start')

    best_valid_score = 0.
    best_epoch =  0

    cam_cnt = 0
    gain_cnt = 0
    cam_total_dice = 0.
    gain_total_dice = 0.

    for idx, (data, target, mask) in enumerate(tqdm(valid_loader, desc="{:17s}".format('Evaluation State'), mininterval=0.01)) :

        if flags['device'] != 'cpu' : data, target = data.to(device), target.to(device)
        
        # Image inverse-normalizing to plot below heatmap
        image = inverse_norm.run(data).numpy()
        image_np = np.transpose(image, (0, 2, 3, 1)).squeeze(0)
        mask_np = mask.numpy().squeeze(0)
        
        baseline_ret, baseline_pred = grad_cam(data, target)
        baseline_ret = upsample(baseline_ret.unsqueeze(0)).detach().cpu()
        baseline_ret = np.transpose(baseline_ret, (0, 2, 3, 1)).squeeze(0)
        
        baseline_threshold = baseline_ret.max()*(0.5)
        baseline_ret = np.where(baseline_ret < baseline_threshold, 0., baseline_ret)

        #_, _, gain_ret, gain_pred = gain_model.attention_map_forward(data, target)
        gain_ret, gain_pred = grad_gain(data, target)
        gain_ret = upsample(gain_ret.unsqueeze(0)).detach().cpu()
        gain_ret = np.transpose(gain_ret, (0, 2, 3, 1)).squeeze(0)
        
        gaine_threshold = gain_ret.max()*(0.5)
        gain_ret = np.where(gain_ret < gaine_threshold, 0., gain_ret)

        if (int(baseline_pred) == int(target)) and (int(target)==1):
            
            try :
                mask_np = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
            except :
                continue
            mask_cnt = cv2.countNonZero(mask_np)

            cam_cnt = cv2.countNonZero(baseline_ret)
            cam_intersect = cv2.countNonZero(cv2.bitwise_and(mask_np, baseline_ret))
            cam_dice = 2 * (cam_intersect / (mask_cnt + cam_cnt))
            cam_total_dice += cam_dice
            
            cam_cnt += 1

        if (int(gain_pred) == int(target)) and (int(target)==1):
            

            try :
                mask_np = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
            except :
                continue
            mask_cnt = cv2.countNonZero(mask_np)

            gain_cnt = cv2.countNonZero(gain_ret)
            gain_intersect = cv2.countNonZero(cv2.bitwise_and(mask_np, gain_ret))
            gain_dice = 2 * (gain_intersect / (mask_cnt + gain_cnt))
            gain_total_dice += gain_dice
            
            gain_cnt += 1
        
    print('gain : ', gain_total_dice / gain_cnt)
    print('cam : ', cam_total_dice / cam_cnt)

        #     figure = plt.figure(figsize = (12, 8))
        #     ax = figure.add_subplot(2, 3, 1)
        #     ax.imshow(image_np)
        #     ax.set_title('Cancer', fontsize = 20)
        #     ax.axis('off')
        #     ax = figure.add_subplot(2, 3, 2)
        #     ax.imshow(image_np)
        #     ax.imshow(baseline_ret, cmap = 'jet', alpha = 0.3)
        #     ax.set_title('Baseline', fontsize = 20)
        #     ax.axis('off')
        #     ax = figure.add_subplot(2, 3, 3)
        #     ax.imshow(image_np)
        #     ax.imshow(gain_ret, cmap = 'jet', alpha = 0.3)
        #     ax.set_title('GAIN', fontsize = 20)
        #     ax.axis('off')
        #     ax = figure.add_subplot(2, 3, 4)
        #     ax.imshow(mask_np)
        #     ax.set_title('Cancer', fontsize = 20)
        #     ax.axis('off')
        #     ax = figure.add_subplot(2, 3, 5)
        #     ax.imshow(mask_np)
        #     ax.imshow(baseline_ret, cmap = 'jet', alpha = 0.3)
        #     ax.set_title('Baseline', fontsize = 20)
        #     ax.axis('off')
        #     ax = figure.add_subplot(2, 3, 6)
        #     ax.imshow(mask_np)
        #     ax.imshow(gain_ret, cmap = 'jet', alpha = 0.3)
        #     ax.set_title('GAIN', fontsize = 20)
        #     ax.axis('off')
        #     #plt.show()
        #     plt.savefig('/home/NAS_mount/sjlee/RHF/export/gain_lung_densenet121/{}.png'.format(str(idx)))
        #     plt.close()
        #     # break

    # for i in range(epoch) :

    #         train_loss, valid_loss = 0.0, 0.0
    #         train_acc, valid_acc = 0.0, 0.0

    #         print('------------[Epoch:{}]-------------'.format(i+1))
    #         gain_model.train()
    #         #cam_model.train()

    #         n_train_batchs = len(train_loader)
    #         n_valid_batchs = len(valid_loader)
    #         batch_size = flags['batch_size']

    #         for train_iter, (samples) in enumerate(tqdm(train_loader, desc="{:17s}".format('Training State'), mininterval=0.01)) :

                
    #             if flags['mask'] : train_data, train_target, _ = samples
    #             else : train_data, train_target = samples

    #             if device != None : train_data, train_target = train_data.to(device), train_target.to(device)
                
    #             total_loss, loss_cl, loss_am, pred, _ = gain_model(train_data, train_target)

    #             total_loss.backward()

    #             gain_model.model.optimizer.step()

                
    #             train_acc += (torch.sum(pred == train_target.data).item()*(100.0 / batch_size))
    #             ts_board.add_scalar('train/total_loss', total_loss.item(), i * n_train_batchs + train_iter)
    #             ts_board.add_scalar('train/cl_loss', loss_cl.item(), i * n_train_batchs + train_iter)
    #             ts_board.add_scalar('train/am_loss', loss_am.item(), i * n_train_batchs + train_iter)

    #             #print(loss_am, loss_cl)

    #             # cam_model.optimizer.zero_grad()

    #             # train_output = cam_model(train_data)

    #             # t_loss = cam_model.loss(train_output, train_target)

    #             # t_loss.backward()

    #             # cam_model.optimizer.step()
                
    #             # _, pred = torch.max(train_output, dim = 1)
            
    #             # train_loss += t_loss.item()
    #             # train_acc += (torch.sum(pred == train_target.data).item()*(100.0 / batch_size))

    #             # ts_board.add_scalar('Loss/train', t_loss.item(), i * n_train_batchs + train_iter)


    #         with torch.enable_grad() :
    #         #with torch.no_grad() :

    #             gain_model.eval()
    #             #cam_model.eval()

    #             for valid_iter, (v_samples) in enumerate(tqdm(valid_loader, desc="{:17s}".format('Evaluation State'), mininterval=0.01)) :

    #                 if flags['mask'] : valid_data, valid_target, _ = v_samples
    #                 else : valid_data, valid_target = v_samples

    #                 if device != None : valid_data, valid_target = valid_data.to(device), valid_target.to(device)

    #                 gain_model.model.optimizer.zero_grad()

    #                 v_total_loss, v_loss_cl, v_loss_am, v_preds, _ = gain_model(valid_data, valid_target)

    #                 valid_acc += (torch.sum(v_preds == valid_target.data).item()*(100.0 / batch_size))
    #                 ts_board.add_scalar('valid/valid_total_loss', v_total_loss.item(), i * n_valid_batchs + valid_iter)
    #                 ts_board.add_scalar('valid/valid_cl_loss', v_loss_cl.item(), i * n_valid_batchs + valid_iter)
    #                 ts_board.add_scalar('valid/valid_am_loss', v_loss_am.item(), i * n_valid_batchs + valid_iter)

    #                 # cam_model.optimizer.zero_grad()

    #                 # valid_output = cam_model(valid_data)

    #                 # v_loss = cam_model.loss(valid_output, valid_target)

    #                 # _, v_pred = torch.max(valid_output, dim = 1)

    #                 # valid_acc += (torch.sum(v_pred == valid_target.data)).item()*(100.0 / batch_size)

    #                 # ts_board.add_scalar('Loss/valid', v_loss.item(), i * n_valid_batchs + valid_iter)


    #         avg_train_acc = train_acc/n_train_batchs
    #         avg_valid_acc = valid_acc/n_valid_batchs

    #         if avg_valid_acc > best_valid_score : 
    #             best_valid_score = avg_valid_acc
    #             best_epoch = i

    #         ts_board.add_scalars('Accuracy', {'train_acc' : avg_train_acc, 
    #                                           'valid_acc' : avg_valid_acc}, i)

    #         curr_lr = gain_model.model.optimizer.param_groups[0]['lr']
    #         #curr_lr = cam_model.optimizer.param_groups[0]['lr']
    #         #training_result = 'epoch.{0:3d} \t train_ac : {1:.4f}% \t  valid_ac : {2:.4f}% \t lr : {3:.6f}\n'.format(i+1, avg_train_acc, avg_valid_acc, curr_lr)

    #         gain_model.model.scheduler.step()
    #         #cam_model.scheduler.step()

    #         epoch_model_params = {
    #             'epoch' : i+1,
    #             'state_dict' : gain_model.state_dict()
    #             #'state_dict' : cam_model.state_dict()
    #         }
    #         torch.save(epoch_model_params, os.path.join(default_path, save_path, save_file+'_{}_epoch.pt'.format(i+1)))
    #         print('epoch.{0:3d} \t train_ac : {1:.4f}% \t  valid_ac : {2:.4f}% \t lr : {3:.6f}\n'.format(i+1, avg_train_acc, avg_valid_acc, curr_lr))
    #         #print('epoch.{0:3d} \t train_ac : {1:.4f}% \t  valid_ac : {2:.4f}% \t lr : {3:.6f}\n'.format(i+1, avg_train_acc, avg_valid_acc, curr_lr))

if __name__ == '__main__' :
    drive(sys.argv[1:])