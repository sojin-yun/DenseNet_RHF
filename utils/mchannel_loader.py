import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class DataLoaderSegmentation(Dataset):
    def __init__(self, root, transform=None, data=None, only_train = True):
        super(DataLoaderSegmentation, self).__init__()
        self.transforms = transform
        self.only_train = only_train

        # fname.png
        self.filenames0 = os.listdir(os.path.join(root,'0'))
        self.filenames1 = os.listdir(os.path.join(root,'1'))

        if data == 'train': 
            self.dir_origin = '/home/NAS_mount/sjyun/Lung_Nodule/Task06_Lung_0315/original/train/0'
            self.dir_origin_mask = '/home/NAS_mount/sjyun/Lung_Nodule/Task06_Lung_0315/original/train_mask/0'
            self.origin_0 = os.listdir(self.dir_origin)
            self.origin_mask0 = os.listdir(self.dir_origin_mask)

        else:
            self.dir_origin = '/home/NAS_mount/sjyun/Lung_Nodule/Task06_Lung_0315/original/val/0'
            self.dir_origin_mask = '/home/NAS_mount/sjyun/Lung_Nodule/Task06_Lung_0315/original/val_mask/0'
            self.origin_0 = os.listdir(self.dir_origin)
            self.origin_mask0 = os.listdir(self.dir_origin_mask)

        # data path
        self.img_files = [os.path.join(root,'0', f0).replace("\\","/") for f0 in self.filenames0 if f0 != 'Thumbs.db'] + [os.path.join(root,'1', f1).replace("\\","/") for f1 in self.filenames1 if f1 != 'Thumbs.db']
        self.mask_files = []
        self.labels = []
        for img_path in self.img_files:
            
            # mask path
            if data == 'train':
                self.mask_files.append(os.path.join(root[:-6],'t_mask', os.path.basename(img_path)).replace("\\","/"))
            else:
                self.mask_files.append(os.path.join(root[:-4],'v_mask', os.path.basename(img_path)).replace("\\","/"))

            # label
            if img_path.split('/')[-2] == '0':
                self.labels.append(0)
            else:
                self.labels.append(1)

    def __getitem__(self, index):
            img_path = self.img_files[index]
            mask_path = self.mask_files[index]
            print(mask_path)
            label = self.labels[index]

            data_fname = img_path.split("/")[-1] #.png
            patient = data_fname.split('img')[0] # pat000
            # slc_num = int(data_fname.split('img')[1].split(".")[0]) # slice num
            slc_num = int(data_fname.split('img')[1].split("_")[0].split(".")[0]) # int(slice num)
                        
            prev_fname = patient + "img" + str(slc_num-1).zfill(5) + ".png"
            next_fname = patient + "img" + str(slc_num+1).zfill(5) + ".png"

            data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if 'r' in data_fname:
                
                ang = data_fname.split("_")[-1].split(".")[0]
                
                if ang == "90":
                    if label == 1:
                        if prev_fname in self.filenames1:
                            prev_data = cv2.imread(img_path[:-23] + prev_fname, cv2.IMREAD_GRAYSCALE)
                            prev_data = cv2.rotate(prev_data, cv2.ROTATE_90_CLOCKWISE)
                            prev_mask = cv2.imread(mask_path[:-23] + prev_fname, cv2.IMREAD_GRAYSCALE)
                            prev_mask = cv2.rotate(prev_mask, cv2.ROTATE_90_CLOCKWISE)
                        else:
                            prev_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                            prev_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                        if next_fname in self.filenames1:
                            next_data = cv2.imread(img_path[:-23] + next_fname, cv2.IMREAD_GRAYSCALE)
                            next_data = cv2.rotate(next_data, cv2.ROTATE_90_CLOCKWISE)
                            next_mask = cv2.imread(mask_path[:-23] + next_fname, cv2.IMREAD_GRAYSCALE)
                            next_mask = cv2.rotate(next_mask, cv2.ROTATE_90_CLOCKWISE)
                        else:
                            next_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                            next_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                    else: # label == 0
                        if prev_fname in self.filenames0:
                            prev_data = cv2.imread(img_path[:-23] + prev_fname, cv2.IMREAD_GRAYSCALE)
                            prev_data = cv2.rotate(prev_data, cv2.ROTATE_90_CLOCKWISE)
                            prev_mask = cv2.imread(mask_path[:-23] + prev_fname, cv2.IMREAD_GRAYSCALE)
                            prev_mask = cv2.rotate(prev_mask, cv2.ROTATE_90_CLOCKWISE)
                        elif prev_fname in self.origin_0:
                            prev_data = cv2.imread(self.dir_origin + "/" + prev_fname, cv2.IMREAD_GRAYSCALE)
                            prev_data = cv2.rotate(prev_data, cv2.ROTATE_90_CLOCKWISE)
                            prev_mask = cv2.imread(self.dir_origin_mask + "/" + prev_fname, cv2.IMREAD_GRAYSCALE)
                            prev_mask = cv2.rotate(prev_mask, cv2.ROTATE_90_CLOCKWISE)
                        else:
                            prev_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                            prev_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                        if next_fname in self.filenames0:
                            next_data = cv2.imread(img_path[:-23] + next_fname, cv2.IMREAD_GRAYSCALE)
                            next_data = cv2.rotate(next_data, cv2.ROTATE_90_CLOCKWISE)
                            next_mask = cv2.imread(mask_path[:-23] + next_fname, cv2.IMREAD_GRAYSCALE)
                            next_mask = cv2.rotate(next_mask, cv2.ROTATE_90_CLOCKWISE)
                        elif next_fname in self.origin_0:
                            next_data = cv2.imread(self.dir_origin + "/" + next_fname, cv2.IMREAD_GRAYSCALE)
                            next_data = cv2.rotate(next_data, cv2.ROTATE_90_CLOCKWISE)
                            next_mask = cv2.imread(self.dir_origin_mask + "/" + next_fname, cv2.IMREAD_GRAYSCALE)
                            next_mask = cv2.rotate(next_mask, cv2.ROTATE_90_CLOCKWISE)
                        else:
                            next_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                            next_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                            
                elif ang == "180":
                    if label == 1:
                        if prev_fname in self.filenames1:
                            prev_data = cv2.imread(img_path[:-24] + prev_fname, cv2.IMREAD_GRAYSCALE)
                            prev_data = cv2.rotate(prev_data, cv2.ROTATE_180)
                            prev_mask = cv2.imread(mask_path[:-24] + prev_fname, cv2.IMREAD_GRAYSCALE)
                            prev_mask = cv2.rotate(prev_mask, cv2.ROTATE_180)
                        else:
                            prev_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                            prev_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                        if next_fname in self.filenames1:
                            next_data = cv2.imread(img_path[:-24] + next_fname, cv2.IMREAD_GRAYSCALE)
                            next_data = cv2.rotate(next_data, cv2.ROTATE_180)
                            next_mask = cv2.imread(mask_path[:-24] + next_fname, cv2.IMREAD_GRAYSCALE)
                            next_mask = cv2.rotate(next_mask, cv2.ROTATE_180)
                        else:
                            next_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                            next_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                    else: # label == 0
                        if prev_fname in self.filenames0:
                            prev_data = cv2.imread(img_path[:-24] + prev_fname, cv2.IMREAD_GRAYSCALE)
                            prev_data = cv2.rotate(prev_data, cv2.ROTATE_180)
                            prev_mask = cv2.imread(mask_path[:-24] + prev_fname, cv2.IMREAD_GRAYSCALE)
                            prev_mask = cv2.rotate(prev_mask, cv2.ROTATE_180)
                        elif prev_fname in self.origin_0:
                            prev_data = cv2.imread(self.dir_origin + "/" + prev_fname, cv2.IMREAD_GRAYSCALE)
                            prev_data = cv2.rotate(prev_data, cv2.ROTATE_180)
                            prev_mask = cv2.imread(self.dir_origin_mask + "/" + prev_fname, cv2.IMREAD_GRAYSCALE)
                            prev_mask = cv2.rotate(prev_mask, cv2.ROTATE_180)
                        else:
                            prev_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                            prev_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                        if next_fname in self.filenames0:
                            next_data = cv2.imread(img_path[:-24] + next_fname, cv2.IMREAD_GRAYSCALE)
                            next_data = cv2.rotate(next_data, cv2.ROTATE_180)
                            next_mask = cv2.imread(mask_path[:-24] + next_fname, cv2.IMREAD_GRAYSCALE)
                            next_mask = cv2.rotate(next_mask, cv2.ROTATE_180)
                        elif next_fname in self.origin_0:
                            next_data = cv2.imread(self.dir_origin + "/" + next_fname, cv2.IMREAD_GRAYSCALE)
                            next_data = cv2.rotate(next_data, cv2.ROTATE_180)
                            next_mask = cv2.imread(self.dir_origin_mask + "/" + next_fname, cv2.IMREAD_GRAYSCALE)
                            next_mask = cv2.rotate(next_mask, cv2.ROTATE_180)
                        else:
                            next_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                            next_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                            
                else:
                    if label == 1:
                        if prev_fname in self.filenames1:
                            prev_data = cv2.imread(img_path[:-24] + prev_fname, cv2.IMREAD_GRAYSCALE)
                            prev_data = cv2.rotate(prev_data, cv2.ROTATE_90_COUNTERCLOCKWISE)
                            prev_mask = cv2.imread(mask_path[:-24] + prev_fname, cv2.IMREAD_GRAYSCALE)
                            prev_mask = cv2.rotate(prev_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        else:
                            prev_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                            prev_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                        if next_fname in self.filenames1:
                            next_data = cv2.imread(img_path[:-24] + next_fname, cv2.IMREAD_GRAYSCALE)
                            next_data = cv2.rotate(next_data, cv2.ROTATE_90_COUNTERCLOCKWISE)
                            next_mask = cv2.imread(mask_path[:-24] + next_fname, cv2.IMREAD_GRAYSCALE)
                            next_mask = cv2.rotate(next_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        else:
                            next_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                            next_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                    else: # label == 0
                        if prev_fname in self.filenames0:
                            prev_data = cv2.imread(img_path[:-24] + prev_fname, cv2.IMREAD_GRAYSCALE)
                            prev_data = cv2.rotate(prev_data, cv2.ROTATE_90_COUNTERCLOCKWISE)
                            prev_mask = cv2.imread(mask_path[:-24] + prev_fname, cv2.IMREAD_GRAYSCALE)
                            prev_mask = cv2.rotate(prev_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        elif prev_fname in self.origin_0:
                            prev_data = cv2.imread(self.dir_origin + "/" + prev_fname, cv2.IMREAD_GRAYSCALE)
                            prev_data = cv2.rotate(prev_data, cv2.ROTATE_90_COUNTERCLOCKWISE)
                            prev_mask = cv2.imread(self.dir_origin_mask + "/" + prev_fname, cv2.IMREAD_GRAYSCALE)
                            prev_mask = cv2.rotate(prev_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        else:
                            prev_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                            prev_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                        if next_fname in self.filenames0:
                            next_data = cv2.imread(img_path[:-24] + next_fname, cv2.IMREAD_GRAYSCALE)
                            next_data = cv2.rotate(next_data, cv2.ROTATE_90_COUNTERCLOCKWISE)
                            next_mask = cv2.imread(mask_path[:-24] + next_fname, cv2.IMREAD_GRAYSCALE)
                            next_mask = cv2.rotate(next_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        elif next_fname in self.origin_0:
                            next_data = cv2.imread(self.dir_origin + "/" + next_fname, cv2.IMREAD_GRAYSCALE)
                            next_data = cv2.rotate(next_data, cv2.ROTATE_90_COUNTERCLOCKWISE)
                            next_mask = cv2.imread(self.dir_origin_mask + "/" + next_fname, cv2.IMREAD_GRAYSCALE)
                            next_mask = cv2.rotate(next_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        else:
                            next_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                            next_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
           
            else:
                if label == 1:
                    if prev_fname in self.filenames1:
                        prev_data = cv2.imread(img_path[:-18] + prev_fname, cv2.IMREAD_GRAYSCALE)
                        prev_mask = cv2.imread(mask_path[:-18] + prev_fname, cv2.IMREAD_GRAYSCALE)
                    else:
                        prev_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        prev_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                    if next_fname in self.filenames1:
                        next_data = cv2.imread(img_path[:-18] + next_fname, cv2.IMREAD_GRAYSCALE)
                        next_mask = cv2.imread(mask_path[:-18] + next_fname, cv2.IMREAD_GRAYSCALE)
                    else:
                        next_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        next_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                else: # label == 0
                    if prev_fname in self.filenames0:
                        prev_data = cv2.imread(img_path[:-18] + prev_fname, cv2.IMREAD_GRAYSCALE)
                        prev_mask = cv2.imread(mask_path[:-18] + prev_fname, cv2.IMREAD_GRAYSCALE)
                    elif prev_fname in self.origin_0:
                        prev_data = cv2.imread(self.dir_origin + "/" + prev_fname, cv2.IMREAD_GRAYSCALE)
                        prev_mask = cv2.imread(self.dir_origin_mask + "/" + prev_fname, cv2.IMREAD_GRAYSCALE)
                    else:
                        prev_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        prev_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                    if next_fname in self.filenames0:
                        next_data = cv2.imread(img_path[:-18] + next_fname, cv2.IMREAD_GRAYSCALE)
                        next_mask = cv2.imread(mask_path[:-18] + next_fname, cv2.IMREAD_GRAYSCALE)
                    elif next_fname in self.origin_0:
                        next_data = cv2.imread(self.dir_origin + "/" + next_fname, cv2.IMREAD_GRAYSCALE)
                        next_mask = cv2.imread(self.dir_origin_mask + "/" + next_fname, cv2.IMREAD_GRAYSCALE)
                    else:
                        next_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        next_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                                    
            
            data = data.reshape(512, 512, 1)
            prev_data = prev_data.reshape(512, 512, 1)
            next_data = next_data.reshape(512, 512, 1)
            mask = mask.reshape(512, 512, 1)
            prev_mask = prev_mask.reshape(512, 512, 1)
            next_mask = next_mask.reshape(512, 512, 1)

            data = np.concatenate((prev_data, data), axis=2)
            data = np.concatenate((data, next_data), axis=2)
            mask = np.concatenate((prev_mask, mask), axis=2)
            mask = np.concatenate((mask, next_mask), axis=2)

            if self.transforms is not None:
                data = self.transforms(data)

            if self.only_train : return data, label
            else : return data, label, mask
                

    def __len__(self):
        return len(self.img_files)