import os, sys
sys.path.append(os.getcwd())
sys.path.append(os.getcwd()+'utils/')
import random
from utils.random_seed import fix_randomness
import shutil


fix_randomness(42)

src_data_path = '/home/NAS_mount/sjlee/ILSVRC/Data/CLS-LOC/train/'
dst_data_path = '/home/NAS_mount/sjlee/Mini_ImageNet/'

total_classes = os.listdir('/home/NAS_mount/sjlee/ILSVRC/Data/CLS-LOC/train/')
sampled_classes = random.sample(total_classes, 100)

for mode in ['train/', 'valid/'] :
    for idx, f in enumerate(sampled_classes) :
        if not os.path.isdir(dst_data_path+mode+f+'/') :
            os.mkdir(dst_data_path+mode+f+'/')
        for n in os.listdir(src_data_path+f+'/') :
            shutil.copyfile(src_data_path+f+'/'+n, dst_data_path+mode+f+'/'+n)
        print(idx/len(sampled_classes))