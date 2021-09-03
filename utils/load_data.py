import os
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

class CustomDataLoader() :
    def __init__(self, args : list) :

        self.args = args
        self.data = args['data']
        self.server = args['server']
        assert self.data in ['cifar100', 'mini_imagenet', 'crc'], '--data argument is invalid'

        self.transformer_components = {
            'cifar100' : {'image_size' : 32, 'mean' : (0.5071, 0.4867, 0.4408), 'std' : (0.2675, 0.2565, 0.2761)},
            'mini_imagenet' : {'image_size' : 224, 'mean' : (0.485, 0.456, 0.406), 'std' : (0.229, 0.224, 0.225)},
            'crc' : None
        }

        self.dataset_components = {
            'cifar100' : {'data' : torchvision.datasets.CIFAR100, 'path' : './data/'},
            'mini_imagenet' : {'data' : datasets.ImageFolder, 'path' : '/home/NAS_mount/sjlee/Mini_ImageNet/'} if self.server else {'data' : datasets.ImageFolder, 'path' : './data/Mini_ImageNet/'},
            'crc' : None
        }

    
    def Transformer(self) :
        image_size = self.transformer_components[self.data]['image_size']
        data_mean, data_std = self.transformer_components[self.data]['mean'], self.transformer_components[self.data]['std']

        train_transformer = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std),
        ])

        valid_transformer = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std)
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

        train_loader = DataLoader(dataset = datasets[0], batch_size = self.args['batch_size'], shuffle = True)
        valid_loader = DataLoader(dataset = datasets[1], batch_size = self.args['batch_size'], shuffle = False)

        return train_loader, valid_loader

    def __call__(self) :
        return self.DataLoader()

