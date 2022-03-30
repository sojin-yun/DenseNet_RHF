import os
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.transforms.transforms import RandomResizedCrop
from utils.module import RandomInversion

class CustomDataLoader() :
    def __init__(self, args : list) :

        self.args = args
        self.data = args['data']
        self.server = args['server']
        assert self.data in ['cifar100', 'mini_imagenet', 'mini_imagenet_vit', 'kidney_stone', 'mnist', 'cub200', 'lung'], '--data argument is invalid'

        self.transformer_components = {
            'cifar100' : {'image_size' : 64, 'mean' : (0.5071, 0.4867, 0.4408), 'std' : (0.2675, 0.2565, 0.2761)},
            'mnist' : {'image_size' : 64, 'mean' : (0.0783, 0.0719, 0.0681), 'std' : (0.2245, 0.2134, 0.2067)},
            'mini_imagenet' : {'image_size' : 224, 'mean' : (0.485, 0.456, 0.406), 'std' : (0.229, 0.224, 0.225)},
            'mini_imagenet_vit' : {'image_size' : 384, 'mean' : (0.485, 0.456, 0.406), 'std' : (0.229, 0.224, 0.225)},
            'kidney_stone' : {'image_size' : 512, 'mean' : 0.161, 'std' : 0.246},
            'cub200' : {'image_size' : 224, 'mean' : (0.485, 0.456, 0.406), 'std' : (0.229, 0.224, 0.225)},
            'lung' : {'image_size' : 512, 'mean' : 0.270, 'std' : 0.309}
        }

        self.dataset_components = {
            'cifar100' : {'data' : torchvision.datasets.CIFAR100, 'path' : '/home/NAS_mount/sjlee/CIFAR100/'} if self.server else {'data' : torchvision.datasets.CIFAR100, 'path' : './data/CIFAR100/'},
            'mnist' : {'data' : datasets.ImageFolder, 'path' : '/home/NAS_mount/sjlee/RHF/data/color_mnist/'} if self.server else {'data' : datasets.ImageFolder, 'path' : './data/color_mnist/'},
            'mini_imagenet' : {'data' : datasets.ImageFolder, 'path' : '/home/NAS_mount/sjlee/RHF/data/Mini_ImageNet/'} if self.server else {'data' : datasets.ImageFolder, 'path' : './data/Mini_ImageNet/'},
            'mini_imagenet_vit' : {'data' : datasets.ImageFolder, 'path' : '/home/NAS_mount/sjlee/RHF/data/Mini_ImageNet/'} if self.server else {'data' : datasets.ImageFolder, 'path' : './data/Mini_ImageNet/'},
            'kidney_stone' : {'data' : datasets.ImageFolder, 'path' : '/home/NAS_mount/sjlee/RHF/data/Kidney_Stone/'} if self.server else {'data' : datasets.ImageFolder, 'path' : './data/Kidney_Stone/'},
            'cub200' : {'data' : datasets.ImageFolder, 'path' : '/home/NAS_mount/sjlee/RHF/data/cub200/'} if self.server else {'data' : datasets.ImageFolder, 'path' : './data/cub200/'},
            'lung' : {'data' : datasets.ImageFolder, 'path' : '/home/NAS_mount/sjlee/RHF/data/Lung_Cancer/'} if self.server else {'data' : datasets.ImageFolder, 'path' : './data/Lung_Cancer/'}
        }

    
    def Transformer(self) :
        self.image_size = self.transformer_components[self.data]['image_size']
        self.data_mean, self.data_std = self.transformer_components[self.data]['mean'], self.transformer_components[self.data]['std']

        if self.data == 'mnist' :
            train_transformer = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomCrop(size = (self.image_size, self.image_size), padding = 8),
                transforms.ToTensor(),
                transforms.Normalize(self.data_mean, self.data_std),
            ])
        else :
            train_transformer = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomCrop(size = (self.image_size, self.image_size), padding = 8),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(self.data_mean, self.data_std),
                #RandomInversion()
            ])

        valid_transformer = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(self.data_mean, self.data_std)
        ])

        return train_transformer, valid_transformer

    def Dataset(self) :
        transforms = self.Transformer()

        if self.data == 'cifar100' :
            train_dataset = self.dataset_components[self.data]['data'](root = self.dataset_components[self.data]['path'], download = True, train = True, transform = transforms[0])
            valid_dataset = self.dataset_components[self.data]['data'](root = self.dataset_components[self.data]['path'], download = True, train = False, transform = transforms[1])
        else :
            train_dataset = self.dataset_components[self.data]['data'](root = os.path.join(self.dataset_components[self.data]['path'], 'train/'), transform = transforms[0])
            valid_dataset = self.dataset_components[self.data]['data'](root = os.path.join(self.dataset_components[self.data]['path'], 'val/'), transform = transforms[1])

        return train_dataset, valid_dataset

    def DataLoader(self) :
        datasets = self.Dataset()

        print('Dataset Information -> {0} || img size : {1} || mean : {2} || std : {3}'.format(self.data, self.image_size, self.data_mean, self.data_std))

        if self.args['mode'] == 'cam' :
            train_loader = None
            valid_loader = DataLoader(dataset = datasets[1], batch_size = 1, shuffle = False, num_workers=0)
        else :
            train_loader = DataLoader(dataset = datasets[0], batch_size = self.args['batch_size'], shuffle = True, num_workers=0)
            valid_loader = DataLoader(dataset = datasets[1], batch_size = self.args['batch_size'], shuffle = True, num_workers=0)

        return train_loader, valid_loader

    def __call__(self) :
        return self.DataLoader()

