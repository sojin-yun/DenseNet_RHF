import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from tqdm import tqdm

class EvaluateMCE() :
    def __init__(self, args, baseline, ensemble, device) :
        self.args = args
        self.device = device
        self.baseline = baseline
        self.ensmeble = ensemble
        self.path = './data/Mini_ImageNet-C'
        self.transforms =  transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.corrupt_list = os.listdir(self.path)

    def load_data(self, corrupt, severity) :
        
        self.dataset = datasets.ImageFolder(os.path.join(self.path, corrupt, str(severity)), self.transforms)
        self.valid_loader = DataLoader(self.dataset, batch_size = self.args['batch_size'], shuffle = True)
        
        return self.valid_loader

    def eval_baseline(self, valid_loader) :

        batch_size = self.args['batch_size']
        valid_loss, valid_acc = 0., 0.

        with torch.no_grad() :

            self.baseline = self.baseline.to(self.device)
            self.baseline.eval()

            for valid_data, valid_target in self.valid_loader :

                if self.device != None : valid_data, valid_target = valid_data.to(self.device), valid_target.to(self.device)

                self.baseline.optimizer.zero_grad()

                valid_output = self.baseline(valid_data)

                v_loss = self.baseline.loss(valid_output, valid_target)

                _, v_pred = torch.max(valid_output, dim = 1)

                valid_loss += v_loss.item()
                valid_acc += (torch.sum(v_pred == valid_target.data)).item()*(100.0 / batch_size)

            avg_valid_acc = valid_acc/len(valid_loader)
            avg_valid_loss = valid_loss/len(valid_loader)

        return avg_valid_acc

    

    def eval_ensemble(self, valid_loader) :
        
        batch_size = self.args['batch_size']
        valid_loss, valid_boundary_loss, valid_ensemble_loss = 0., 0., 0.
        valid_acc, valid_boundary_acc, valid_ensemble_acc = 0., 0., 0.

        with torch.no_grad() :

            self.ensmeble = self.ensmeble.to(self.device)
            self.ensmeble.eval()

            for valid_data, valid_target in self.valid_loader :

                if self.device != None : valid_data, valid_target = valid_data.to(self.device), valid_target.to(self.device)

                self.ensmeble.optimizer.zero_grad()

                valid_output, valid_boundary_output, valid_ensemble_output = self.ensmeble(valid_data)

                v_loss = self.ensmeble.loss(valid_output, valid_target)
                valid_b_loss = self.ensmeble.boundary_loss(valid_boundary_output, valid_target)
                valid_e_loss = self.ensmeble.ensemble_loss(valid_ensemble_output, valid_target)

                _, v_pred = torch.max(valid_output, dim = 1)
                _, v_boundary_pred = torch.max(valid_boundary_output, dim = 1)
                _, v_ensemble_pred = torch.max(valid_ensemble_output, dim = 1)

                valid_loss += v_loss.item()
                valid_boundary_loss += valid_b_loss.item()
                valid_ensemble_loss += valid_e_loss.item()
                valid_acc += (torch.sum(v_pred == valid_target.data)).item()*(100.0 / batch_size)
                valid_boundary_acc += (torch.sum(v_boundary_pred == valid_target.data)).item()*(100.0 / batch_size)
                valid_ensemble_acc += (torch.sum(v_ensemble_pred == valid_target.data)).item()*(100.0 / batch_size)

            avg_valid_acc = valid_acc/len(valid_loader)
            avg_boundary_valid_acc = valid_boundary_acc/len(valid_loader)
            avg_ensemble_valid_acc = valid_ensemble_acc/len(valid_loader)

        return avg_valid_acc, avg_boundary_valid_acc, avg_ensemble_valid_acc

    def run(self) :
        for c in self.corrupt_list :
            print('Evaluation on corruption-{}'.format(c))
            baseline_ret = 0.
            ensemble_ret = [0., 0., 0.]
            for s in tqdm(range(1, 6), desc="{:17s}".format('Evaluation State'), mininterval=0.01) :
                data_loader = self.load_data(c, str(s))
                baseline_ret += self.eval_baseline(data_loader)
                ret = list(self.eval_ensemble(data_loader))
                ensemble_ret[0] += ret[0]
                ensemble_ret[1] += ret[1]
                ensemble_ret[2] += ret[2]
            print('\ncorruption-{}'.format(c))
            print('Baseline : {:.4f}%'.format(baseline_ret/5.))
            print('Ensemble : {:.4f}% | {:.4f}% | {:.4f}%'.format(ensemble_ret[0]/5., ensemble_ret[1]/5., ensemble_ret[2]/5.))

            