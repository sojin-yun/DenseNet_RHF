from pytorch_pretrained_vit import ViT

def VisionTransformer(type : str, pretrained : bool = True) :
    return ViT(type, pretrained)