import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os
import torch
import torch.nn as nn
from utils.grad_cam import GradCAM
from utils.select_model import Select_Model
from utils.image_process import InverseNormalize
from torch.utils.data import DataLoader

class RunGradCAM() :
    def __init__(self, args, data_loader) :

        self.args = args
        n_device = 'cuda:{}'.format(self.args['device']) if self.args['device'] != 'cpu' else 'cpu'
        self.device = torch.device(n_device)
        _, self.test_loader = data_loader
        self.default_path = '/home/NAS_mount/sjlee/RHF/' if self.args['server'] else '.'
        self.save_path = os.path.join(self.default_path, 'export/', self.args['dst'])
        if not os.path.isdir(self.save_path) :
            os.mkdir(self.save_path)
            os.mkdir(os.path.join(self.save_path, 'Both_correct'))
            os.mkdir(os.path.join(self.save_path, 'Ensemble_correct'))

        self.image_size = {'mini_imagenet' : 224, 'cifar100' : 64, 'kidney_stone' : 512, 'lung' : 512}
        self.mean = {'mini_imagenet' : (0.485, 0.456, 0.406), 'cifar100' : (0.5071, 0.4867, 0.4408), 'kidney_stone' : (0.161, 0.161, 0.161), 'lung' : (0.270, 0.270, 0.270)}
        self.std = {'mini_imagenet' : (0.229, 0.224, 0.225), 'cifar100' : (0.2675, 0.2565, 0.2761), 'kidney_stone' : (0.246, 0.246, 0.246), 'lung' : (0.309, 0.309, 0.309)}

        self._make_folder()
        self._make_model()
        # for i, m in enumerate(list(self.baseline_model.modules())) :
        #     print('---------------------------', i, '--------------------------')
        #     print(m)
        # return

    def _make_model(self) :

        self.baseline_model = Select_Model(args = self.args, device = self.device).baseline_model(model = self.args['model'])
        self.ensemble_model = Select_Model(args = self.args, device = self.device).ensemble_model(model = self.args['model'])

        baseline_params = torch.load('{0}/weights/evaluation/{1}'.format(self.default_path, self.args['cam'][0]), map_location = self.device)
        ensemble_params = torch.load('{0}/weights/evaluation/{1}'.format(self.default_path, self.args['cam'][1]), map_location = self.device)

        self.baseline_model.load_state_dict(baseline_params['state_dict'])
        self.ensemble_model.load_state_dict(ensemble_params['state_dict'])

        # [mini_imagenet, cifar100]
        self.hooked_layer = {'baseline' : {'vgg16' : [44, 43], 'vgg19' : [53, 32], 'resnet50' : [152, 152], 'resnet101' : [288, 287], 'resnet152' : [424, 423], 'densenet121' : [492, 491], 'densenet169' : [684, 683], 'densenet201' : [812, 811], 'rensenet' : [180, 180], 'srensenet' : [116, 116]},
                             'ensemble' : {'vgg16' : [142, 102], 'vgg19' : [141, 111], 'resnet50' : [201, 198], 'resnet101' : [322, 321], 'resnet152' : [458, 457], 'densenet121' : [528, 518], 'densenet169' : [717, 716], 'densenet201' : [845, 844], 'rensenet' : [225, 225], 'srensenet' : [160, 160]}}
        idx = 1 if self.args['data'] == 'cifar100' else 0

        self.baseline_cam = GradCAM(model = self.baseline_model, hooked_layer = self.hooked_layer['baseline'][self.args['model']][idx], device = self.device, ensemble = False)
        self.ensemble_cam = GradCAM(model = self.ensemble_model, hooked_layer = self.hooked_layer['ensemble'][self.args['model']][idx], device = self.device, ensemble = True)

    def _make_folder(self) :

        n_class = 2 if (self.args['data'] == 'kidney_stone' or self.args['data'] == 'lung') else 100
        for ret_type in ['Both_correct/', 'Ensemble_correct/'] :
            for idx in range(n_class) :
                if not os.path.isdir(os.path.join(self.default_path, self.save_path, ret_type, str(idx))) :
                    os.mkdir(os.path.join(self.default_path, self.save_path, ret_type, str(idx)))

    def run(self) :

        #image_size = 224 if self.args['data'] == 'mini_imagenet' else 64
        image_size = self.image_size[self.args['data']]
        mean, std = self.mean[self.args['data']], self.std[self.args['data']]
        #mean, std = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) if self.args['data'] == 'mini_imagenet' else ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        self.upsample = nn.Upsample(size = image_size, mode = 'bilinear', align_corners = False)
        
        inverse_norm = InverseNormalize(mean, std)

        if self.args['data'] == 'kidney_stone' :
            mapping_dict = {'0' : 'No Stone', '1' : 'Stone'}
        elif self.args['data'] == 'lung' :
            mapping_dict = {'0' : 'No Cancer', '1' : 'Cancer'}
        else :
            dict_name = 'cifar_dict.json' if self.args['data'] == 'cifar100' else 'mini_dict.json'
            dict_path = '/home/NAS_mount/sjlee/RHF/data/{}'.format(dict_name) if self.args['server'] else 'data/{}'.format(dict_name)
            load_dict = open(dict_path, 'r')
            mapping_dict = json.load(load_dict)

        baseline_result = np.empty((len(self.test_loader), image_size, image_size, 1))
        ensemble_result = np.empty((len(self.test_loader), image_size, image_size, 1))

        for idx, (data, target) in enumerate(tqdm(self.test_loader, desc="{:17s}".format('Evaluation State'), mininterval=0.01)) :

            if self.args['device'] != 'cpu' : data, target = data.to(self.device), target.to(self.device)
            
            # Image inverse-normalizing to plot below heatmap
            image = inverse_norm.run(data).numpy()
            image_np = np.transpose(image, (0, 2, 3, 1)).squeeze(0)
            
            baseline_ret, baseline_pred = self.baseline_cam(data, target)
            baseline_ret = self.upsample(baseline_ret.unsqueeze(0)).detach().cpu()
            baseline_ret = np.transpose(baseline_ret, (0, 2, 3, 1)).squeeze(0)
            
            baseline_threshold = baseline_ret.max()*(0.5)
            baseline_ret = np.where(baseline_ret < baseline_threshold, 0., baseline_ret)
            
            baseline_result[idx] = baseline_ret
            
            ensemble_ret, ensemble_pred = self.ensemble_cam(data, target)
            ensemble_ret = self.upsample(ensemble_ret.unsqueeze(0)).detach().cpu()
            ensemble_ret = np.transpose(ensemble_ret, (0, 2, 3, 1)).squeeze(0)
            
            ensemble_threshold = ensemble_ret.max()*(0.5)
            ensemble_ret = np.where(ensemble_ret < ensemble_threshold, 0., ensemble_ret)

            ensemble_result[idx] = ensemble_ret

            figsave_path = ''

            if (baseline_pred.item() != target.item()) and (ensemble_pred.item() == target.item()) : figsave_path = os.path.join(self.default_path, self.save_path, 'Ensemble_correct/', str(target.item()))
            elif (baseline_pred.item() == target.item()) and (ensemble_pred.item() == target.item()) : figsave_path = os.path.join(self.default_path, self.save_path, 'Both_correct/', str(target.item()))
            else : continue

            fig = plt.figure(figsize=(12, 4))
            ax0 = fig.add_subplot(1, 3, 1)
            ax0.imshow((image_np * 255.).astype('uint8'))
            ax0.set_title(mapping_dict[str(target.item())], fontsize = 18)
            #ax0.set_title(target.item(), fontsize = 15)
            ax0.axis('off')

            ax1 = fig.add_subplot(1, 3, 2)
            ax1.imshow((image_np * 255.).astype('uint8'))
            ax1.imshow((baseline_ret * 255.).astype('uint8'), cmap = 'jet', alpha = 0.4)
            ax1.set_title('Baseline', fontsize = 15)
            ax1.axis('off')

            ax2 = fig.add_subplot(1, 3, 3)
            ax2.imshow((image_np * 255.).astype('uint8'))
            ax2.imshow((ensemble_ret * 255.).astype('uint8'), cmap = 'jet', alpha = 0.4)
            ax2.set_title('Ensemble', fontsize = 15)
            ax2.axis('off')

            plt.savefig(figsave_path+'/{0}.png'.format(str(idx)))
            plt.close()
            #plt.show()

        np.save(os.path.join(self.default_path, self.save_path, 'baseline_cam.npy'), baseline_result)
        np.save(os.path.join(self.default_path, self.save_path, 'ensemble_cam.npy'), ensemble_result)

    def run_with_mask(self) :

        #image_size = 224 if self.args['data'] == 'mini_imagenet' else 64
        image_size = self.image_size[self.args['data']]
        mean, std = self.mean[self.args['data']], self.std[self.args['data']]
        #mean, std = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) if self.args['data'] == 'mini_imagenet' else ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        self.upsample = nn.Upsample(size = image_size, mode = 'bilinear', align_corners = False)
        
        inverse_norm = InverseNormalize(mean, std)

        if self.args['data'] == 'kidney_stone' :
            mapping_dict = {'0' : 'No Stone', '1' : 'Stone'}
        elif self.args['data'] == 'lung' :
            mapping_dict = {'0' : 'No Cancer', '1' : 'Cancer'}
        else :
            dict_name = 'cifar_dict.json' if self.args['data'] == 'cifar100' else 'mini_dict.json'
            dict_path = '/home/NAS_mount/sjlee/RHF/data/{}'.format(dict_name) if self.args['server'] else 'data/{}'.format(dict_name)
            load_dict = open(dict_path, 'r')
            mapping_dict = json.load(load_dict)

        baseline_result = np.empty((len(self.test_loader), image_size, image_size, 1))
        ensemble_result = np.empty((len(self.test_loader), image_size, image_size, 1))

        for idx, (data, target, mask) in enumerate(tqdm(self.test_loader, desc="{:17s}".format('Evaluation State'), mininterval=0.01)) :

            if self.args['device'] != 'cpu' : data, target = data.to(self.device), target.to(self.device)
            
            # Image inverse-normalizing to plot below heatmap
            image = inverse_norm.run(data).numpy()
            image_np = np.transpose(image, (0, 2, 3, 1)).squeeze(0)
            mask_np = mask.numpy().squeeze(0)
            
            baseline_ret, baseline_pred = self.baseline_cam(data, target)
            baseline_ret = self.upsample(baseline_ret.unsqueeze(0)).detach().cpu()
            baseline_ret = np.transpose(baseline_ret, (0, 2, 3, 1)).squeeze(0)
            
            baseline_threshold = baseline_ret.max()*(0.5)
            baseline_ret = np.where(baseline_ret < baseline_threshold, 0., baseline_ret)
            
            baseline_result[idx] = baseline_ret
            
            ensemble_ret, ensemble_pred = self.ensemble_cam(data, target)
            ensemble_ret = self.upsample(ensemble_ret.unsqueeze(0)).detach().cpu()
            ensemble_ret = np.transpose(ensemble_ret, (0, 2, 3, 1)).squeeze(0)
            
            ensemble_threshold = ensemble_ret.max()*(0.5)
            ensemble_ret = np.where(ensemble_ret < ensemble_threshold, 0., ensemble_ret)

            ensemble_result[idx] = ensemble_ret

            figsave_path = ''

            if (baseline_pred.item() != target.item()) and (ensemble_pred.item() == target.item()) : figsave_path = os.path.join(self.default_path, self.save_path, 'Ensemble_correct/', str(target.item()))
            elif (baseline_pred.item() == target.item()) and (ensemble_pred.item() == target.item()) : figsave_path = os.path.join(self.default_path, self.save_path, 'Both_correct/', str(target.item()))
            else : continue

            fig = plt.figure(figsize=(12, 8))
            ax0 = fig.add_subplot(2, 3, 1)
            ax0.imshow((image_np * 255.).astype('uint8'))
            ax0.set_title(mapping_dict[str(target.item())], fontsize = 18)
            #ax0.set_title(target.item(), fontsize = 15)
            ax0.axis('off')

            ax1 = fig.add_subplot(2, 3, 2)
            ax1.imshow((image_np * 255.).astype('uint8'))
            ax1.imshow((baseline_ret * 255.).astype('uint8'), cmap = 'jet', alpha = 0.4)
            ax1.set_title('Baseline', fontsize = 15)
            ax1.axis('off')

            ax2 = fig.add_subplot(2, 3, 3)
            ax2.imshow((image_np * 255.).astype('uint8'))
            ax2.imshow((ensemble_ret * 255.).astype('uint8'), cmap = 'jet', alpha = 0.4)
            ax2.set_title('Ensemble', fontsize = 15)
            ax2.axis('off')

            ax3 = fig.add_subplot(2, 3, 4)
            ax3.imshow(mask_np, cmap = 'gray')
            ax3.set_title(mapping_dict[str(target.item())], fontsize = 18)
            #ax3.set_title(target.item(), fontsize = 15)
            ax3.axis('off')

            ax4 = fig.add_subplot(2, 3, 5)
            ax4.imshow(mask_np, cmap = 'gray')
            ax4.imshow((baseline_ret * 255.).astype('uint8'), cmap = 'jet', alpha = 0.4)
            ax4.set_title('120th epoch / Acc : 71.42%', fontsize = 12)
            ax4.axis('off')

            ax5 = fig.add_subplot(2, 3, 6)
            ax5.imshow(mask_np, cmap = 'gray')
            ax5.imshow((ensemble_ret * 255.).astype('uint8'), cmap = 'jet', alpha = 0.4)
            ax5.set_title('105th epoch / Acc : 73.41%', fontsize = 12)
            ax5.axis('off')

            plt.savefig(figsave_path+'/{0}.png'.format(str(idx)))
            plt.close()
            #plt.show()

        np.save(os.path.join(self.default_path, self.save_path, 'baseline_cam.npy'), baseline_result)
        np.save(os.path.join(self.default_path, self.save_path, 'ensemble_cam.npy'), ensemble_result)

    def run_separated(self) :

        #image_size = 224 if self.args['data'] == 'mini_imagenet' else 64
        image_size = self.image_size[self.args['data']]
        mean, std = self.mean[self.args['data']], self.std[self.args['data']]
        #mean, std = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) if self.args['data'] == 'mini_imagenet' else ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        self.upsample = nn.Upsample(size = image_size, mode = 'bilinear', align_corners = False)
        
        inverse_norm = InverseNormalize(mean, std)

        if self.args['data'] == 'kidney_stone' :
            mapping_dict = {'0' : 'No Stone', '1' : 'Stone'}
        else :
            dict_name = 'cifar_dict.json' if self.args['data'] == 'cifar100' else 'mini_dict.json'
            dict_path = '/home/NAS_mount/sjlee/RHF/data/{}'.format(dict_name) if self.args['server'] else 'data/{}'.format(dict_name)
            load_dict = open(dict_path, 'r')
            mapping_dict = json.load(load_dict)

        baseline_result = np.empty((len(self.test_loader), image_size, image_size, 1))
        ensemble_result = np.empty((len(self.test_loader), image_size, image_size, 1))

        for idx, (data, target) in enumerate(tqdm(self.test_loader, desc="{:17s}".format('Evaluation State'), mininterval=0.01)) :

            if self.args['device'] != 'cpu' : data, target = data.to(self.device), target.to(self.device)
            
            # Image inverse-normalizing to plot below heatmap
            image = inverse_norm.run(data).numpy()
            image_np = np.transpose(image, (0, 2, 3, 1)).squeeze(0)
            
            baseline_ret, baseline_pred = self.baseline_cam(data, target)
            baseline_ret = self.upsample(baseline_ret.unsqueeze(0)).detach().cpu()
            baseline_ret = np.transpose(baseline_ret, (0, 2, 3, 1)).squeeze(0)
            baseline_result[idx] = baseline_ret
            
            ensemble_ret, ensemble_ret_backbone, ensemble_ret_boundary, ensemble_pred = self.ensemble_cam.separated_forward(data, target)
            ensemble_ret = self.upsample(ensemble_ret.unsqueeze(0)).detach().cpu()
            ensemble_ret_backbone = self.upsample(ensemble_ret_backbone.unsqueeze(0)).detach().cpu()
            ensemble_ret_boundary = self.upsample(ensemble_ret_boundary.unsqueeze(0)).detach().cpu()
            ensemble_ret = np.transpose(ensemble_ret, (0, 2, 3, 1)).squeeze(0)
            ensemble_ret_backbone = np.transpose(ensemble_ret_backbone, (0, 2, 3, 1)).squeeze(0)
            ensemble_ret_boundary = np.transpose(ensemble_ret_boundary, (0, 2, 3, 1)).squeeze(0)
            ensemble_result[idx] = ensemble_ret_backbone

            figsave_path = ''

            if (baseline_pred.item() != target.item()) and (ensemble_pred.item() == target.item()) : figsave_path = os.path.join(self.default_path, self.save_path, 'Ensemble_correct/', str(target.item()))
            elif (baseline_pred.item() == target.item()) and (ensemble_pred.item() == target.item()) : figsave_path = os.path.join(self.default_path, self.save_path, 'Both_correct/', str(target.item()))
            else : continue

            fig = plt.figure(figsize=(20, 4))
            ax0 = fig.add_subplot(1, 5, 1)
            ax0.imshow(image_np)
            ax0.set_title(mapping_dict[str(target.item())], fontsize = 18)
            #ax0.set_title(target.item(), fontsize = 15)
            ax0.axis('off')

            ax1 = fig.add_subplot(1, 5, 2)
            ax1.imshow(image_np)
            ax1.imshow(baseline_ret, cmap = 'jet', alpha = 0.4)
            ax1.set_title('Baseline', fontsize = 15)
            ax1.axis('off')

            ax2 = fig.add_subplot(1, 5, 3)
            ax2.imshow(image_np)
            ax2.imshow(ensemble_ret, cmap = 'jet', alpha = 0.4)
            ax2.set_title('Ensemble', fontsize = 15)
            ax2.axis('off')

            ax3 = fig.add_subplot(1, 5, 4)
            ax3.imshow(image_np)
            ax3.imshow(ensemble_ret_backbone, cmap = 'jet', alpha = 0.4)
            ax3.set_title('Ensemble-backbone', fontsize = 15)
            ax3.axis('off')

            ax4 = fig.add_subplot(1, 5, 5)
            ax4.imshow(image_np)
            ax4.imshow(ensemble_ret_boundary, cmap = 'jet', alpha = 0.4)
            ax4.set_title('Ensemble-edge', fontsize = 15)
            ax4.axis('off')

            plt.savefig(figsave_path+'/{0}.png'.format(str(idx)))
            plt.close()
            #plt.show()

        np.save(os.path.join(self.default_path, self.save_path, 'baseline_cam.npy'), baseline_result)
        np.save(os.path.join(self.default_path, self.save_path, 'ensemble_cam.npy'), ensemble_result)

