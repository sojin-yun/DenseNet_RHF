import os, sys
sys.path.append(os.getcwd()+'utils/')
import random
from utils.random_seed import fix_randomness

fix_randomness(42)

total_classes = os.listdir('/home/NAS_mount/sjlee/ILSVRC/Data/CLS-LOC/train/')
sampled_classes = random.sample(total_classes, 100)
sampled_classes.sort()
print(sampled_classes)

