import os, sys
import random
from utils.random_seed import fix_randomness
from utils.append_path import add_sys_path

fix_randomness(42)

add_sys_path(os.getcwd())

total_classes = os.listdir('/home/NAS_mount/sjlee/ILSVRC/Data/CLS-LOC/train/')
sampled_classes = random.sample(total_classes, 100)
sampled_classes.sort()
print(sampled_classes)

