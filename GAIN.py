from re import S
from tkinter import image_names
import torch
import torch.nn as nn
import torch.nn.functional as F

class GAIN(nn.Module):
    def __init__(self,  device, model, hooked_layer):
        super(GAIN, self).__init__()
        self.device = device
        self.model = model

        self.forward_result = None
        self.backward_result = None

        self.loss = nn.CrossEntropyLoss()

        #total_loss = loss_cl + alpha*loss_am
        self.alpha = 0.01
        
        #for softmask
        self.sigma = 0.7
        self.omega = 1e5

        #hook
        list(self.model.modules())[hooked_layer].register_forward_hook(self.forward_hook)
        print('Hook on {0}'.format(list(self.model.modules())[hooked_layer]))
        
    def forward_hook(self, _, input_image, output):
        input_image[0].register_hook(self.hook)
        self.forward_result = torch.squeeze(output) 


    def hook(self, grad):   #backward hook
        self.backward_result = grad     
        
    

    def attention_map_forward(self, images, labels):
        #generate gradcam
        output_cl = self.model(images)
        Sc = 0.
        _, predictions = torch.max(output_cl, 1)
        for idx in range(len(labels)): #for each image batch
            _, pred = torch.max(output_cl[idx], dim = 0) #_: max value, pred : the index of the max value
            # predictions.append(pred.item())
            Sc += output_cl[idx, pred]  #loss에 crossentropy가 아닌, y^c자체를 줌 (y^c를 feature map으로 미분하기 위해)
        
        #back propagation
        Sc.backward(retain_graph=True)
        self.model.zero_grad()

        fl = F.relu(self.forward_result)
        loss_cl = self.loss(output_cl, labels)
        pool = nn.AdaptiveAvgPool2d((1, 1))
        weights = pool(self.backward_result)
        Ac = torch.mul(fl, weights).sum(dim=1, keepdim=True)
        Ac = F.relu(Ac)
        Ac = F.upsample_bilinear(Ac, size=images.size()[2:])
        return output_cl, loss_cl, Ac, predictions

    def softmask(self, Ac, images):
        Ac_min = torch.amin(Ac, dim = (1, 2, 3), keepdim=True)
        Ac_max = torch.amax(Ac, dim = (1, 2, 3), keepdim=True)
        Ac_min = Ac.min()
        Ac_max = Ac.max()
        try:
            scaled_ac = (Ac - Ac_min) / (Ac_max - Ac_min)
        except ZeroDivisionError:
            scaled_ac = torch.zeros_like(Ac)
        mask = torch.sigmoid(self.omega * (scaled_ac - self.sigma))
        masked_image = images - images * mask
        return masked_image

    
    def forward(self, images, labels):
        output_cl, loss_cl, Ac, preds = self.attention_map_forward(images, labels)

        I_star = self.softmask(Ac, images) 
        output_am = self.model(I_star)
        self.model.zero_grad()
        # ImageNet
        # loss_am = torch.sum(torch.sigmoid(output_am), dim=(0,1), keepdim=False)
        # loss_am /= len(labels)
        # Lung
        loss_am = torch.sum(torch.softmax(output_am), dim=(0,1), keepdim=False)
        loss_am /= len(labels)
        total_loss = loss_cl + self.alpha * loss_am
        return total_loss, loss_cl, loss_am, Ac, preds

        





        




