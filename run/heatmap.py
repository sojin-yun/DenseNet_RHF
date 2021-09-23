import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from utils.grad_cam import GradCAM
from utils.select_model import Select_Model
from torch.utils.data import DataLoader

class RunGradCAM() :
    def __init__(self, args, data_loader) :

        self.args = args
        n_device = 'cuda:{}'.format(self.args['device']) if self.args['device'] != 'cpu' else 'cpu'
        self.device = torch.device(n_device)
        _, self.test_loader = data_loader
        self.default_path = '/home/NAS_mount/sjlee/RHF/' if self.args['server'] else '.'
        self.save_path = os.path.join('export/', self.default_path, self.args['dst'])
        if not os.path.isdir(self.save_path) :
            os.mkdir(self.save_path)

        self._make_model()

    def _make_model(self) :

        self.baseline_model = Select_Model(args = self.args, device = self.device).baseline_model(model = self.args['model'])
        self.ensemble_model = Select_Model(args = self.args, device = self.device).ensemble_model(model = self.args['model'])

        baseline_params = torch.load('{0}/weights/evaluation/{1}'.format(self.default_path, self.args['cam'][0]), map_location = self.device)
        ensemble_params = torch.load('{0}/weights/evaluation/{1}'.format(self.default_path, self.args['cam'][1]), map_location = self.device)

        self.baseline_model.load_state_dict(baseline_params['state_dict'])
        self.ensemble_model.load_state_dict(ensemble_params['state_dict'])

        # [mini_imagenet, cifar100]
        self.hooked_layer = {'baseline' : {'vgg16' : [44, 43], 'vgg19' : [53, 32], 'resnet50' : [152, 151], 'resnet101' : [288, 287], 'resnet152' : [424, 423], 'densenet121' : [492, 491], 'densenet169' : [684, 683], 'densenet201' : [812, 811]},
                             'ensemble' : {'vgg16' : [132, 102], 'vgg19' : [141, 111], 'resnet50' : [186, 185], 'resnet101' : [322, 321], 'resnet152' : [458, 457], 'densenet121' : [525, 524], 'densenet169' : [717, 716], 'densenet201' : [845, 844]}}
        idx = 0 if self.args['data'] == 'mini_imagenet' else 1
        print(self.hooked_layer['baseline'][self.args['model']][idx])
        self.baseline_cam = GradCAM(model = self.baseline_model, hooked_layer = self.hooked_layer['baseline'][self.args['model']][idx])
        self.ensemble_model = GradCAM(model = self.ensemble_model, hooked_layer = self.hooked_layer['ensemble'][self.args['model']][idx])

    def run(self) :

        image_size = 224 if self.args['data'] == 'mini_imagenet' else 64
        mean, std = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) if self.args['data'] == 'mini_imagenet' else ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        self.upsample = nn.Upsample(size = image_size, mode = 'bilinear', align_corners = False)
        
        self.baseline_model.eval()
        self.ensemble_model.eval()

        with torch.no_grad():
            for idx, (data, target) in enumerate(tqdm(self.test_loader, desc="{:17s}".format('Evaluation State'), mininterval=0.01)) :
                pass
