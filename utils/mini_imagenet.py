import os, sys
sys.path.append(os.getcwd())
sys.path.append(os.getcwd()+'utils/')
import random
from utils.random_seed import fix_randomness


fix_randomness(42)

src_data_path = '/home/NAS_mount/sjlee/ILSVRC/Data/CLS-LOC/train/'
dst_data_path = '/home/NAS_mount/sjlee/Mini_ImageNet/'

total_classes = os.listdir('/home/NAS_mount/sjlee/ILSVRC/Data/CLS-LOC/train/')
sampled_classes = random.sample(total_classes, 100)

for f in sampled_classes :
    print(f)