from pytorch_pretrained_vit import ViT
import torch.nn as nn

def VisionTransformer(type : str, pretrained : bool = True) :
    model = ViT(type, pretrained)
    for n, m in model.named_modeles() :
        if n == 'fc' :
            setattr(model, n, nn.Linear(768, 100, bias = True))
    return model