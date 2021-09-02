import os, sys
sys.path.append(os.getcwd())
sys.path.append(os.getcwd()+'utils/')
import random
from utils.random_seed import fix_randomness
import shutil
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as image


fix_randomness(42)

src_data_path = '/home/NAS_mount/sjlee/ILSVRC/Data/CLS-LOC'
dst_data_path = '/home/NAS_mount/sjlee/Mini_ImageNet/'

if not os.path.isdir(dst_data_path) :
    os.mkdir(dst_data_path)

total_classes = os.listdir('/home/NAS_mount/sjlee/ILSVRC/Data/CLS-LOC/train/')
sampled_classes = random.sample(total_classes, 100)
sampled_classes.sort()

# for mode in ['train/', 'val/'] :
#     for idx, f in enumerate(sampled_classes) :
#         if not os.path.isdir(dst_data_path+mode+f+'/') :
#             os.mkdir(dst_data_path+mode+f+'/')
#         for n in os.listdir(src_data_path+mode+f+'/') :
#             shutil.copyfile(src_data_path+mode+f+'/'+n, dst_data_path+mode+f+'/'+n)
#         print(idx/len(sampled_classes))

class_txt = open(dst_data_path+'class_label.txt', 'w')
for f in sampled_classes :
    class_txt.write(f+'\n')
class_txt.close()

summary_class = plt.figure(figsize=(20, 20))
visualization_path = os.path.join(src_data_path, 'val/')
print(visualization_path)
for idx, cls in os.listdir(visualization_path) :
    print(idx, cls)
    ax = summary_class.add_subplot(10, 10, idx)
    file_name = os.listdir(os.path.join(visualization_path, cls))
    img = image.open(os.path.join(src_data_path, 'val/', cls))
    ax.imshow(img)
    ax.axis('off')
summary_class.savefig('class_sample.png')
