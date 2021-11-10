import torch
from torch._C import BoolType
import torch.nn as nn

class GradCAM(nn.Module) :
    def __init__(self, model, hooked_layer, device, ensemble : BoolType) :
        super(GradCAM, self).__init__()
        
        # hook on target layer.
        self.model = model.to(device)
        self.ensemble = ensemble
        self.model_type = 'ensemble' if self.ensemble else 'baseline'
        list(self.model.modules())[hooked_layer].register_forward_hook(self.forward_hook)
        print('In {}'.format(self.model_type))
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

    def separated_forward(self, image_batch, label_batch):

        assert self.model_type == 'ensemble', 'Network type is not ensemble!'

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

        self.forward_backbone = self.forward_result[:512, :, :]
        self.forward_boundary = self.forward_result[512:, :, :]
        self.backbone_result = self.backward_result[:, :512, :, :]
        self.boundary_result = self.backward_result[:, 512:, :, :]
        print(self.backward_result.shape)
        print(self.forward_result.shape)

        if len(self.backward_result.shape) == 3:
            a_k = torch.mean(self.backbone_result.unsqueeze(0), dim=(2, 3), keepdim=True)
            a_k_backbone = torch.mean(self.backward_result.unsqueeze(0), dim=(2, 3), keepdim=True)
            a_k_boundary = torch.mean(self.boundary_result.unsqueeze(0), dim=(2, 3), keepdim=True)
        else:
            a_k = torch.mean(self.backward_result, dim=(2, 3), keepdim=True)
            a_k_backbone = torch.mean(self.backbone_result, dim=(2, 3), keepdim=True)
            a_k_boundary = torch.mean(self.boundary_result, dim=(2, 3), keepdim=True)

        cam = torch.sum(a_k * torch.nn.functional.relu(self.forward_result), dim=1)
        cam_backbone = torch.sum(a_k_backbone * torch.nn.functional.relu(self.forward_backbone), dim=1)
        cam_boundary = torch.sum(a_k_boundary * torch.nn.functional.relu(self.forward_boundary), dim=1)

        cam_relu = torch.nn.functional.relu(cam)
        cam_relu_backbone = torch.nn.functional.relu(cam_backbone)
        cam_relu_boundary = torch.nn.functional.relu(cam_boundary)

        return cam_relu, cam_relu_backbone, cam_relu_boundary, pred
