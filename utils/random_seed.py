import random
import torch
import numpy as np

def Fix_Randomness(in_number) :
    seed_number = in_number
    #torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed_number)
    torch.cuda.manual_seed(seed_number)
    torch.cuda.manual_seed_all(seed_number)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_number)
    random.seed(seed_number)
