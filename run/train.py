import torch
from torch.utils import data
from tqdm import tqdm
import os
import time
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import copy

class TrainingEnsemble :

    def __init__(self, args, model, data_loader, device = None) :

        self.args = args
        self.epoch = args['epoch']
        self.save_path = args['dst']
        self.save_file = args['file']
        self.model = model
        self.device = device
        self.train_loader, self.valid_loader = data_loader
        if self.args['data'] == 'cifar100' : 
            self.default_path = '/home/NAS_mount/sjlee/RHF/Save_parameters/cifar100/' if self.args['server'] else './Save_parameters/cifar100/'
            self.model_size = (3, 64, 64)
        elif self.args['data'] == 'mnist' : 
            self.default_path = '/home/NAS_mount/sjlee/RHF/Save_parameters/mnist/' if self.args['server'] else './Save_parameters/mnist/'
            self.model_size = (3, 64, 64)
        elif self.args['data'] == 'mini_imagenet' : 
            self.default_path = '/home/NAS_mount/sjlee/RHF/Save_parameters/mini_imagenet/' if self.args['server'] else './Save_parameters/mini_imagenet/'
            self.model_size = (3, 224, 224)
        elif self.args['data'] == 'mini_imagenet_vit' : 
            self.default_path = '/home/NAS_mount/sjlee/RHF/Save_parameters/mini_imagenet/' if self.args['server'] else './Save_parameters/mini_imagenet/'
            self.model_size = (3, 384, 384)
        elif self.args['data'] == 'kidney_stone' : 
            self.default_path = '/home/NAS_mount/sjlee/RHF/Save_parameters/kidney_stone/' if self.args['server'] else './Save_parameters/kidney_stone/'
            self.model_size = (1, 512, 512)

        if self.device != None : 
            self.device = device
            self.model = self.model.to(self.device)

    def __call__(self) :
        self.run()

    def run(self) :
        
        # Log hyper-parameters, model summary and tensorboard
        if not os.path.isdir(os.path.join(self.default_path, self.save_path)) :
            os.mkdir(os.path.join(self.default_path, self.save_path))
        
        if self.args['tensorboard'] : 
            folder = 'tensor_board'
            os.mkdir(os.path.join(self.default_path, self.save_path, folder))
            self.ts_board = SummaryWriter(log_dir = os.path.join(self.default_path, self.save_path, folder))

        if not self.args['server'] :
            s = open(os.path.join(self.default_path, self.save_path, 'model_summary.txt'), 'w')
            s.write('Model : {}-ensemble model. \n\n'.format(self.args['model']))
            # copy_model = copy.deepcopy(self.model)
            # copy_model.to('cpu')
            # model_summary = summary(copy_model, self.model_size, batch_size = 1, device = 'cpu')
            # del copy_model
            # torch.cuda.empty_cache()
            # for l in model_summary :
            #     s.write(l+'\n')
            s.close()

        print('\n\nMake model_summary.txt and log.txt')
        f = open(os.path.join(self.default_path, self.save_path, 'log.txt'), 'w')
        print(os.path.join(self.default_path, self.save_path, 'log.txt'))
        now = time.localtime()
        f.write("Start training at {:04d}/{:02d}/{:02d}--{:02d}:{:02d}:{:02d}\n\n".format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))
        f.write("Argument Information : {}\n\n".format(self.args))
        if self.args['device'] == 'cpu' :
            f.write('GPU Information - CPU\n\n')
        else :
            f.write('GPU Information - {}\n\n'.format(torch.cuda.get_device_name('cuda:{}'.format(self.args['device']))))
        print('Make log.txt and log training result\n\n')

        if self.args['device'] == 'cpu' :
            print('GPU Information - CPU\n\n')
        else :
            print('GPU Information - {}\n\n'.format(torch.cuda.get_device_name('cuda:{}'.format(self.args['device']))))

        best_valid_acc, best_boundary_valid_acc, best_ensemble_valid_acc = 0., 0., 0.

        # for non-pretrained
        #backbone_loss_weight, boundary_loss_weight, ensemble_loss_weight = 1.0, 0.2, 0.2
        # for pre-trained
        backbone_loss_weight, boundary_loss_weight, ensemble_loss_weight = 1.0, 0.5, 0.5
        f.write('Loss Information - Backbone_loss : {0} | Boundary_loss : {1} | Ensemble_loss : {2}\n\n'.format(backbone_loss_weight, boundary_loss_weight, ensemble_loss_weight))
        f.write('Optimizer : {}\n'.format(self.model.optimizer))
        f.write('Learning_scheduler : step_size : {0} | gamma : {1}\n\n'.format(self.model.scheduler.step_size, self.model.scheduler.gamma))
        for i in range(self.epoch) :

            train_loss, valid_loss = 0.0, 0.0
            train_acc, valid_acc = 0.0, 0.0
            boundary_loss, valid_boundary_loss = 0.0, 0.0
            boundary_acc, valid_boundary_acc = 0.0, 0.0
            ensemble_loss, valid_ensemble_loss = 0.0, 0.0
            ensemble_acc, valid_ensemble_acc = 0.0, 0.0

            print('------------[Epoch:{}]-------------'.format(i+1))
            self.model.train()

            n_train_batchs = len(self.train_loader)
            n_valid_batchs = len(self.valid_loader)
            batch_size = self.args['batch_size']

            for train_iter, (train_data, train_target) in enumerate(tqdm(self.train_loader, desc="{:17s}".format('Training State'), mininterval=0.01)) :
                
                if self.device != None : train_data, train_target = train_data.to(self.device), train_target.to(self.device)
                
                self.model.optimizer.zero_grad()

                train_output, boundary_output, ensemble_output = self.model(train_data)

                t_loss = self.model.loss(train_output, train_target)
                b_loss = self.model.boundary_loss(boundary_output, train_target)
                e_loss = self.model.ensemble_loss(ensemble_output, train_target)

                sum_loss = (t_loss*(backbone_loss_weight) + b_loss*(boundary_loss_weight) + e_loss*(ensemble_loss_weight))
                sum_loss.backward()

                self.model.optimizer.step()
                
                _, pred = torch.max(train_output, dim = 1)
                _, boundary_pred = torch.max(boundary_output, dim = 1)
                _, ensemble_pred = torch.max(ensemble_output, dim = 1)

            
                train_loss += t_loss.item()
                boundary_loss += b_loss.item()
                ensemble_loss += e_loss.item()
                train_acc += (torch.sum(pred == train_target.data).item()*(100.0 / batch_size))
                boundary_acc += (torch.sum(boundary_pred == train_target.data).item()*(100.0 / batch_size))
                ensemble_acc += (torch.sum(ensemble_pred == train_target.data).item()*(100.0 / batch_size))

                if self.args['tensorboard'] :
                    self.ts_board.add_scalars('Loss/train', {'t_loss' : t_loss.item(), 
                                                             't_boundary_loss' : b_loss.item(), 
                                                             't_ensemble_loss' : e_loss.item()}, i * n_train_batchs + train_iter)

            with torch.no_grad() :

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

                    valid_loss += v_loss.item()
                    valid_boundary_loss += valid_b_loss.item()
                    valid_ensemble_loss += valid_e_loss.item()
                    valid_acc += (torch.sum(v_pred == valid_target.data)).item()*(100.0 / batch_size)
                    valid_boundary_acc += (torch.sum(v_boundary_pred == valid_target.data)).item()*(100.0 / batch_size)
                    valid_ensemble_acc += (torch.sum(v_ensemble_pred == valid_target.data)).item()*(100.0 / batch_size)

                    if self.args['tensorboard'] :
                        self.ts_board.add_scalars('Loss/valid', {'v_loss' : v_loss.item(), 
                                                                 'v_boundary_loss' : valid_b_loss.item(), 
                                                                 'v_ensemble_loss' : valid_e_loss.item()}, i * n_valid_batchs + valid_iter)


            avg_train_acc = train_acc/n_train_batchs
            avg_boundary_train_acc = boundary_acc/n_train_batchs
            avg_ensemble_train_acc = ensemble_acc/n_train_batchs
            avg_valid_acc = valid_acc/n_valid_batchs
            avg_boundary_valid_acc = valid_boundary_acc/n_valid_batchs
            avg_ensemble_valid_acc = valid_ensemble_acc/n_valid_batchs
            avg_tarin_loss = ensemble_loss/n_train_batchs
            avg_valid_loss = valid_ensemble_loss/n_train_batchs

            if self.args['tensorboard'] :
                    self.ts_board.add_scalars('Accuracy', {'train_acc' : avg_ensemble_train_acc, 
                                                           'valid_acc' : avg_ensemble_valid_acc}, i)
                    self.ts_board.add_scalars('Loss_per_epoch', {'train_loss' : avg_tarin_loss, 
                                                                 'valid_loss' : avg_valid_loss}, i)

            curr_lr = self.model.optimizer.param_groups[0]['lr']

            self.model.scheduler.step()

            if avg_valid_acc > best_valid_acc : best_valid_acc = avg_valid_acc
            if avg_boundary_valid_acc > best_boundary_valid_acc : best_boundary_valid_acc = avg_boundary_valid_acc
            if avg_ensemble_valid_acc > best_ensemble_valid_acc : 
                best_ensemble_valid_acc = avg_ensemble_valid_acc
                best_model_params = {
                'epoch' : i,
                'state_dict' : self.model.state_dict(),
                'optimizer' : self.model.optimizer.state_dict(),
                'scheduler' : self.model.scheduler.state_dict()
                }
                torch.save(best_model_params, os.path.join(self.default_path, self.save_path, self.save_file+'.pt'))

            training_result = 'epoch.{0:3d} \t train_ac : {1:.4f}% \t  valid_ac : {2:.4f}% \t bdr_train : {3:.4f}% \t bdr_valid : {4:.4f}% \t ens_train : {5:.4f}% \t ens_valid : {6:.4f}% \t lr : {7:.6f}\n'.format(i+1, avg_train_acc, avg_valid_acc, avg_boundary_train_acc, avg_boundary_valid_acc, avg_ensemble_train_acc, avg_ensemble_valid_acc, curr_lr)
            f.write(training_result)
            print(training_result)

        # Training is finished.
        f.write('\nBest valid acc : {0:.4f}% \t Best boundary acc : {1:.4f}% \t Best ensemble acc : {2:.4f}%\n'.format(best_valid_acc, best_boundary_valid_acc, best_ensemble_valid_acc))
        f.close()
        print('Best valid acc : {0:.4f}% \t Best boundary acc : {1:.4f}% \t Best ensemble acc : {2:.4f}%'.format(best_valid_acc, best_boundary_valid_acc, best_ensemble_valid_acc))
        print("Argument Information : {}\n\n".format(self.args))

class TrainingBaseline :

    def __init__(self, args, model, data_loader, device = None) :

        self.args = args
        self.epoch = args['epoch']
        self.save_path = args['dst']
        self.save_file = args['file']
        self.model = model
        self.device = device
        self.train_loader, self.valid_loader = data_loader
        if self.args['data'] == 'cifar100' : 
            self.default_path = '/home/NAS_mount/sjlee/RHF/Save_parameters/cifar100/' if self.args['server'] else './Save_parameters/cifar100/'
            self.model_size = (3, 64, 64)
        elif self.args['data'] == 'mini_imagenet' : 
            self.default_path = '/home/NAS_mount/sjlee/RHF/Save_parameters/mini_imagenet/' if self.args['server'] else './Save_parameters/mini_imagenet/'
            self.model_size = (3, 224, 224)
        elif self.args['data'] == 'mini_imagenet_vit' : 
            self.default_path = '/home/NAS_mount/sjlee/RHF/Save_parameters/mini_imagenet/' if self.args['server'] else './Save_parameters/mini_imagenet/'
            self.model_size = (3, 384, 384)
        elif self.args['data'] == 'kidney_stone' : 
            self.default_path = '/home/NAS_mount/sjlee/RHF/Save_parameters/kidney_stone/' if self.args['server'] else './Save_parameters/kidney_stone/'
            self.model_size = (1, 512, 512)

        if self.device != None : 
            self.device = device
            self.model = self.model.to(self.device)

    def __call__(self) :
        self.run()

    def run(self) :
        
        # Log hyper-parameters, model summary and tensorboard
        if not os.path.isdir(os.path.join(self.default_path, self.save_path)) :
            os.mkdir(os.path.join(self.default_path, self.save_path))

        if self.args['tensorboard'] : 
            folder = 'tensor_board'
            os.mkdir(os.path.join(self.default_path, self.save_path, folder))
            self.ts_board = SummaryWriter(log_dir = os.path.join(self.default_path, self.save_path, folder))

        if not self.args['server'] and self.args['data'] != 'kidney_stone' :
            s = open(os.path.join(self.default_path, self.save_path, 'model_summary.txt'), 'w')
            s.write('Model : {}-baseline model. \n\n'.format(self.args['model']))
            copy_model = copy.deepcopy(self.model)
            copy_model.to('cpu')
            model_summary = summary(copy_model, self.model_size, batch_size = 1, device = 'cpu')
            del copy_model
            torch.cuda.empty_cache()
            for l in model_summary :
                s.write(l+'\n')
            s.close()

        print('\n\nMake model_summary.txt and log.txt')
        f = open(os.path.join(self.default_path, self.save_path, 'log.txt'), 'w')
        print(os.path.join(self.default_path, self.save_path, 'log.txt'))
        now = time.localtime()
        f.write("Start training at {:04d}/{:02d}/{:02d}--{:02d}:{:02d}:{:02d}\n\n".format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))
        f.write("Argument Information : {}\n\n".format(self.args))
        if self.args['device'] == 'cpu' :
            f.write('GPU Information - cpu\n\n')
        else :
            f.write('GPU Information - {}\n\n'.format(torch.cuda.get_device_name('cuda:{}'.format(self.args['device']))))
        print('Make log.txt and log training result\n\n')
        if self.args['device'] == 'cpu' :
            print('GPU Information - cpu\n\n')
        else :
            print('GPU Information - {}\n\n'.format(torch.cuda.get_device_name('cuda:{}'.format(self.args['device']))))
        f.write('Optimizer : {}\n'.format(self.model.optimizer))
        f.write('Learning_scheduler : step_size : {0} | gamma : {1}\n\n'.format(self.model.scheduler.step_size, self.model.scheduler.gamma))

        best_valid_acc, best_valid_loss = 0., 100.


        for i in range(self.epoch) :

            train_loss, valid_loss = 0.0, 0.0
            train_acc, valid_acc = 0.0, 0.0

            print('------------[Epoch:{}]-------------'.format(i+1))
            self.model.train()

            n_train_batchs = len(self.train_loader)
            n_valid_batchs = len(self.valid_loader)
            batch_size = self.args['batch_size']

            for train_iter, (train_data, train_target) in enumerate(tqdm(self.train_loader, desc="{:17s}".format('Training State'), mininterval=0.01)) :
                
                if self.device != None : train_data, train_target = train_data.to(self.device), train_target.to(self.device)
                
                self.model.optimizer.zero_grad()

                train_output = self.model(train_data)

                t_loss = self.model.loss(train_output, train_target)

                t_loss.backward()

                self.model.optimizer.step()
                
                _, pred = torch.max(train_output, dim = 1)
            
                train_loss += t_loss.item()
                train_acc += (torch.sum(pred == train_target.data).item()*(100.0 / batch_size))

                if self.args['tensorboard'] :
                    self.ts_board.add_scalar('Loss/train', t_loss.item(), i * n_train_batchs + train_iter)

            with torch.no_grad() :

                self.model.eval()

                for valid_iter, (valid_data, valid_target) in enumerate(tqdm(self.valid_loader, desc="{:17s}".format('Evaluation State'), mininterval=0.01)) :

                    if self.device != None : valid_data, valid_target = valid_data.to(self.device), valid_target.to(self.device)

                    self.model.optimizer.zero_grad()

                    valid_output = self.model(valid_data)

                    v_loss = self.model.loss(valid_output, valid_target)

                    _, v_pred = torch.max(valid_output, dim = 1)

                    valid_loss += v_loss.item()
                    valid_acc += (torch.sum(v_pred == valid_target.data)).item()*(100.0 / batch_size)

                    if self.args['tensorboard'] :
                        self.ts_board.add_scalar('Loss/valid', v_loss.item(), i * n_valid_batchs + valid_iter)

            avg_train_acc = train_acc/n_train_batchs
            avg_valid_acc = valid_acc/n_valid_batchs
            avg_tarin_loss = train_loss/n_train_batchs
            avg_valid_loss = valid_loss/n_valid_batchs

            if self.args['tensorboard'] :
                self.ts_board.add_scalars('Accuracy', {'train_acc' : avg_train_acc, 
                                                       'valid_acc' : avg_valid_acc}, i)
                self.ts_board.add_scalars('Loss_per_epoch', {'train_loss' : avg_tarin_loss, 
                                                             'valid_loss' : avg_valid_loss}, i)

            curr_lr = self.model.optimizer.param_groups[0]['lr']

            self.model.scheduler.step()

            if avg_valid_acc > best_valid_acc : 
                best_model_params = {
                'epoch' : i,
                'state_dict' : self.model.state_dict(),
                'optimizer' : self.model.optimizer.state_dict(), 
                'scheduler' : self.model.scheduler.state_dict()
                }
                best_valid_acc = avg_valid_acc
                best_valid_loss = avg_valid_loss
                torch.save(best_model_params, os.path.join(self.default_path, self.save_path, self.save_file+'.pt'))
            
            training_result = 'epoch.{0:3d} \t train_ac : {1:.4f}% \t  valid_ac : {2:.4f}% \t lr : {3:.6f}\n'.format(i+1, avg_train_acc, avg_valid_acc, curr_lr)
            f.write(training_result)
            print(training_result)

        # Training is finished.
        f.write('\nBest valid acc : {0:.4f}% \t Best valid loss : {1:.6f} \n'.format(best_valid_acc, best_valid_loss))
        f.close()
        print('Best valid acc : {0:.4f}% \t Best valid loss : {1:.6f} \n\n'.format(best_valid_acc, best_valid_loss))
        print("Argument Information : {}\n\n".format(self.args))
