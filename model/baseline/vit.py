from pytorch_pretrained_vit import ViT
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, LambdaLR

def VisionTransformer(type : str, pretrained : bool = True) :
    model = ViT(type, pretrained)

    optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9, weight_decay=0.0015)
    #optimizer = optim.SGD(model.parameters(), lr = 8e-4, momentum = 0.9, weight_decay=0.1)
    setattr(model, 'optimizer', optimizer)

    loss = nn.CrossEntropyLoss()
    setattr(model, 'loss', loss)
    
    scheduler = StepLR(model.optimizer, step_size=15, gamma=0.5)
    #scheduler = LambdaLR(model.optimizer, lr_lambda=lambda epoch:0.95**epoch)
    setattr(model, 'scheduler', scheduler)

    modified_fc = nn.Linear(768, 100, bias = True)
    nn.init.xavier_normal_(modified_fc.weight)

    for n, m in model.named_modules() :
        if n == 'fc' :
            setattr(model, n, modified_fc)
    return model