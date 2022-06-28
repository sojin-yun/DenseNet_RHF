from pytorch_pretrained_vit import ViT
#from vit_pytorch import ViT
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, StepLR, LambdaLR, CosineAnnealingLR

def VisionTransformer(image_size : int, num_classes : int) :

    #model = ViT(name = 'B_16', pretrained = False, image_size = 512, num_classes = 2, num_heads = 16, num_layers = 6, ff_dim = 2048, patches = 64, dim = 1024)
    model = ViT(name = 'B_16_imagenet1k', pretrained = True, image_size = 512, num_classes = 2)

    optimizer = optim.SGD(model.parameters(), lr = 1e-2, momentum = 0.9, weight_decay=0.0001)

    setattr(model, 'optimizer', optimizer)

    loss = nn.CrossEntropyLoss()
    setattr(model, 'loss', loss)
    
    #scheduler = MultiStepLR(model.optimizer, milestones=[100, 150], gamma=0.1)
    scheduler = CosineAnnealingLR(model.optimizer, T_max = 50, eta_min = 0)
    setattr(model, 'scheduler', scheduler)

    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)

    return model

# def VisionTransformer(image_size : int, num_classes : int) :
    
#     model = ViT(
#         image_size = image_size,
#         patch_size = 64,
#         num_classes = num_classes,
#         dim = 1024,
#         depth = 6,
#         heads = 16,
#         mlp_dim = 2048,
#         dropout = 0.1,
#         emb_dropout = 0.1
#     )  

#     optimizer = optim.SGD(model.parameters(), lr = 1e-3, momentum = 0.9, weight_decay=0.0001)

#     setattr(model, 'optimizer', optimizer)

#     loss = nn.CrossEntropyLoss()
#     setattr(model, 'loss', loss)
    
#     scheduler = MultiStepLR(model.optimizer, milestones=[100, 150], gamma=0.1)
#     #scheduler = LambdaLR(model.optimizer, lr_lambda=lambda epoch:0.95**epoch)
#     setattr(model, 'scheduler', scheduler)

#     for m in model.modules():
#         if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
#             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#         elif isinstance(m, nn.BatchNorm2d):
#             nn.init.constant_(m.weight, 1)
#             nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.Linear):
#             nn.init.xavier_normal_(m.weight)

#     return model