import torch
from torch._C import BoolType
import torch.nn as nn

class GradCAM(nn.Module) :
    def __init__(self, model, hooked_layer, ensemble : BoolType) :
        super(GradCAM, self).__init__()
        
        # hook on target layer.
        self.model = model
        self.ensemble = ensemble
        list(self.model.modules())[hooked_layer].register_forward_hook(self.forward_hook)
        print('Hook on {0}'.format(list(self.model.modules())[hooked_layer]))

    def forward_hook(self, _, input_image, output):
        input_image[0].register_hook(self.hook)
        self.forward_result = torch.squeeze(output)

    def hook(self, grad):
        self.backward_result = grad

    def forward(self, image_batch, label_batch):
        
        self.model.eval()

        if self.ensemble :
            _, _, output = self.model(image_batch)
        else :
            output = self.model(image_batch)

        loss = 0.
        for idx in range(len(label_batch)):
            _, pred = torch.max(output[idx], dim = 0)
            loss += output[idx, pred]
        
        loss.backward()

        if len(self.backward_result.shape) == 3:
            a_k = torch.mean(self.backward_result.unsqueeze(0), dim=(2, 3), keepdim=True)
        else:
            a_k = torch.mean(self.backward_result, dim=(2, 3), keepdim=True)
        cam = torch.sum(a_k * torch.nn.functional.relu(self.forward_result), dim=1)
        cam_relu = torch.nn.functional.relu(cam)

        return cam_relu, pred
