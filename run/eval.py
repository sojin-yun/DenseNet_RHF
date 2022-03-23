import torch
import torch.nn.functional as F
from torch.utils import data
from tqdm import tqdm
import os
import time
from torchsummary import summary
from sklearn.metrics import f1_score, precision_score, recall_score, PrecisionRecallDisplay
import numpy as np
import matplotlib.pyplot as plt

class Evaluation :

    def __init__(self, args, model, data_loader, device = None) :
        self.args = args
        self.model = model
        self.device = device
        _, self.valid_loader = data_loader

    def __call__(self) :
        if self.args['baseline'] : self.run_baseline()
        else : self.run_ensemble()

    def run_ensemble(self) :
        
        batch_size = self.args['batch_size']
        valid_loss, valid_boundary_loss, valid_ensemble_loss = 0., 0., 0.
        valid_acc, valid_boundary_acc, valid_ensemble_acc = 0., 0., 0.
        total_precision, total_recall, total_f1score = 0., 0., 0.

        with torch.no_grad() :

            self.model = self.model.to(self.device)
            self.model.eval()

            for valid_iter, (valid_data, valid_target) in enumerate(tqdm(self.valid_loader, desc="{:17s}".format('Evaluation State'), mininterval=0.01)) :

                if self.device != None : valid_data, valid_target = valid_data.to(self.device), valid_target.to(self.device)

                self.model.optimizer.zero_grad()

                valid_output, valid_boundary_output, valid_ensemble_output = self.model(valid_data)

                v_loss = self.model.loss(valid_output, valid_target)
                valid_b_loss = self.model.boundary_loss(valid_boundary_output, valid_target)
                valid_e_loss = self.model.ensemble_loss(valid_ensemble_output, valid_target)

                _, v_pred = torch.max(valid_output, dim = 1)
                _, v_boundary_pred = torch.max(valid_boundary_output, dim = 1)
                _, v_ensemble_pred = torch.max(valid_ensemble_output, dim = 1)

                np_target, np_pred = np.array(valid_target.cpu()), np.array(v_ensemble_pred.cpu())
                total_precision += precision_score(np_target, np_pred)
                total_recall += recall_score(np_target, np_pred)
                total_f1score += f1_score(np_target, np_pred)

                valid_loss += v_loss.item()
                valid_boundary_loss += valid_b_loss.item()
                valid_ensemble_loss += valid_e_loss.item()
                valid_acc += (torch.sum(v_pred == valid_target.data)).item()*(100.0 / batch_size)
                valid_boundary_acc += (torch.sum(v_boundary_pred == valid_target.data)).item()*(100.0 / batch_size)
                valid_ensemble_acc += (torch.sum(v_ensemble_pred == valid_target.data)).item()*(100.0 / batch_size)

            avg_valid_acc = valid_acc/len(self.valid_loader)
            avg_boundary_valid_acc = valid_boundary_acc/len(self.valid_loader)
            avg_ensemble_valid_acc = valid_ensemble_acc/len(self.valid_loader)
            precision, recall, f1score = total_precision/len(self.valid_loader)*100., total_recall/len(self.valid_loader)*100., total_f1score/len(self.valid_loader)*100.

        print('\n\nEvaluation Result --- Backbone Acc : {0:.4f}% | Boundary Acc : {1:.4f}% | Ensemble Acc : {2:.4f}%'.format(avg_valid_acc, avg_boundary_valid_acc, avg_ensemble_valid_acc))
        print('\n\nEvaluation Result --- Ensemble Precision : {0:.4f}% | Ensemble Recall : {1:.4f}% | Ensemble F1-Score : {2:.4f}%'.format(precision, recall, f1score))


    def run_baseline(self) :
        
        batch_size = self.args['batch_size']
        valid_loss, valid_acc = 0., 0.
        total_precision, total_recall, total_f1score = 0., 0., 0.

        with torch.no_grad() :

            self.model = self.model.to(self.device)
            self.model.eval()

            for valid_iter, (valid_data, valid_target) in enumerate(tqdm(self.valid_loader, desc="{:17s}".format('Evaluation State'), mininterval=0.01)) :

                if self.device != None : valid_data, valid_target = valid_data.to(self.device), valid_target.to(self.device)

                self.model.optimizer.zero_grad()

                valid_output = self.model(valid_data)

                v_loss = self.model.loss(valid_output, valid_target)

                _, v_pred = torch.max(valid_output, dim = 1)

                np_target, np_pred = np.array(valid_target.cpu()), np.array(v_pred.cpu())
                total_precision += precision_score(np_target, np_pred)
                total_recall += recall_score(np_target, np_pred)
                total_f1score += f1_score(np_target, np_pred)

                valid_loss += v_loss.item()
                valid_acc += (torch.sum(v_pred == valid_target.data)).item()*(100.0 / batch_size)


            avg_valid_acc = valid_acc/len(self.valid_loader)
            avg_valid_loss = valid_loss/len(self.valid_loader)
            precision, recall, f1score = total_precision/len(self.valid_loader)*100., total_recall/len(self.valid_loader)*100., total_f1score/len(self.valid_loader)*100.

        print('\n\nEvaluation Result --- Valid Loss : {0:.4f} | Valid Acc : {1:.4f}%'.format(avg_valid_loss, avg_valid_acc))
        print('\n\nEvaluation Result --- Baseline Precision : {0:.4f}% | Baseline Recall : {1:.4f}% | Baseline F1-Score : {2:.4f}%'.format(precision, recall, f1score))

class AveragePrecision :
    def __init__(self, args, model, data_loader, n_threshold, device = None) :
        self.args = args
        self.model = model
        self.device = device
        self.n_threshold = n_threshold
        self.bins = [0.1*(i+1) for i in range(n_threshold)]
        _, self.valid_loader = data_loader

    def __call__(self) :
        if self.args['baseline'] : self.run_baseline()
        else : self.run_ensemble()

    def run_ensemble(self) :

        batch_size = self.args['batch_size']
        precision, recall, f1score = 0., 0., 0.
        total_pr, total_rc, total_f1 = [], [], []

        with torch.no_grad() :

            self.model = self.model.to(self.device)
            self.model.eval()

            for threshold in self.bins :

                target, prediction = None, None

                for valid_data, valid_target in self.valid_loader :

                    if self.device != None : valid_data, valid_target = valid_data.to(self.device), valid_target.to(self.device)

                    self.model.optimizer.zero_grad()

                    _, _, valid_ensemble_output = self.model(valid_data)
                    valid_ensemble_output = F.softmax(valid_ensemble_output, dim = 1)

                    #_, v_ensemble_pred = torch.max(valid_ensemble_output, dim = 1)
                    valid_ensemble_output = valid_ensemble_output[:, 1]
                    valid_ensemble_output = torch.where(valid_ensemble_output >= threshold, 1, 0)

                    if target == None : target = valid_target
                    else : target = torch.cat([target, valid_target], dim = 0)

                    if prediction == None : prediction = valid_ensemble_output
                    else : prediction = torch.cat([prediction, valid_ensemble_output], dim = 0)

                np_target, np_pred = np.array(target.cpu()), np.array(prediction.cpu())

                precision = precision_score(np_target, np_pred)
                recall = recall_score(np_target, np_pred)
                f1score = f1_score(np_target, np_pred)

                print('\nAbout threshold [{3:.1f}] --- ensemble Precision : {0:.4f}% | ensemble Recall : {1:.4f}% | ensemble F1-Score : {2:.4f}%'.format(precision*100., recall*100., f1score*100., threshold))

                total_pr.append(precision)
                total_rc.append(recall)
                total_f1.append(f1score)

            total_pr, total_rc, total_f1 = np.array(total_pr), np.array(total_rc), np.array(total_f1)
            
            plt.plot(total_pr, total_rc)
            plt.show()

    def run_baseline(self) :

        batch_size = self.args['batch_size']
        precision, recall, f1score = 0., 0., 0.
        total_pr, total_rc, total_f1 = [], [], []

        with torch.no_grad() :

            self.model = self.model.to(self.device)
            self.model.eval()

            for threshold in self.bins :

                target, prediction = None, None

                for valid_data, valid_target in self.valid_loader :

                    if self.device != None : valid_data, valid_target = valid_data.to(self.device), valid_target.to(self.device)

                    self.model.optimizer.zero_grad()

                    valid_output = self.model(valid_data)
                    valid_output = F.softmax(valid_output, dim = 1)

                    #_, v_ensemble_pred = torch.max(valid_ensemble_output, dim = 1)
                    valid_output = valid_output[:, 1]
                    valid_output = torch.where(valid_output >= threshold, 1, 0)

                    if target == None : target = valid_target
                    else : target = torch.cat([target, valid_target], dim = 0)

                    if prediction == None : prediction = valid_output
                    else : prediction = torch.cat([prediction, valid_output], dim = 0)

                np_target, np_pred = np.array(target.cpu()), np.array(prediction.cpu())

                precision = precision_score(np_target, np_pred)
                recall = recall_score(np_target, np_pred)
                f1score = f1_score(np_target, np_pred)

                print('\nAbout threshold [{3:.1f}] --- ensemble Precision : {0:.4f}% | ensemble Recall : {1:.4f}% | ensemble F1-Score : {2:.4f}%'.format(precision*100., recall*100., f1score*100., threshold))

                total_pr.append(precision)
                total_rc.append(recall)
                total_f1.append(f1score)

            total_pr, total_rc, total_f1 = np.array(total_pr), np.array(total_rc), np.array(total_f1)
            
            plt.plot(total_pr, total_rc)
            plt.show()
