import torch
import torch.nn as nn
import torch.nn.functional as F

class ReviveMaxPool2d(nn.Module) :
    def __init__(self, kernel_size, stride = None, padding = 0, dilation = 1, return_indices: bool = False, ceil_mode: bool = False, upsample_size = None) -> None:
        super(ReviveMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        
        self.max_pool2d = nn.MaxPool2d(kernel_size = self.kernel_size, stride = self.stride, padding = self.padding, dilation = self.dilation, 
                                       return_indices = self.return_indices, ceil_mode = self.ceil_mode)
        if upsample_size == None :
            self.upsample = nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=False)
        else :
            self.upsample = nn.Upsample(size = upsample_size, mode = 'bilinear', align_corners=False)
        self.recovered_map = None
        
    def forward(self, x) :
        # Init recovered map
        self.recovered_map = None

        # Calculate boundary map
        out = self.max_pool2d(x)
        up = self.upsample(out)
        self.recovered_map = torch.abs((x-up))
        return out

class ReviveAvgPool2d(nn.Module) :
    def __init__(self, kernel_size, stride = None, padding = 0, ceil_mode: bool = False, count_include_pad: bool = True, upsample_size = None) -> None:
        super(ReviveAvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        
        self.avg_pool2d = nn.AvgPool2d(kernel_size = self.kernel_size, stride = self.stride, padding = self.padding, ceil_mode = self.ceil_mode, count_include_pad = self.count_include_pad)
        if upsample_size == None :
            self.upsample = nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=False)
        else :
            self.upsample = nn.Upsample(size = upsample_size, mode = 'bilinear', align_corners=False)
        self.recovered_map = None
        
    def forward(self, x) :
        # Init recovered map
        self.recovered_map = None
        
        # Calculate boundary map
        out = self.avg_pool2d(x)
        up = self.upsample(out)
        self.recovered_map = torch.abs((x-up))
        return out


class ReviveConv2d(nn.Module) :
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride = 1,
                 padding = 0, dilation = 1, groups: int = 1,
                 bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None, batch_norm = True) -> None :
        super(ReviveConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        self.device = device
        self.dtype = dtype
        self.conv = nn.Conv2d(in_channels = self.in_channels, out_channels = self.out_channels, kernel_size = self.kernel_size, stride = 2, padding = self.padding,
                                    dilation = self.dilation, groups = self.groups, bias = self.bias, padding_mode = self.padding_mode, device = self.device, dtype = self.dtype)
        if batch_norm :
            self.dsize_bn = nn.BatchNorm2d(self.out_channels)
            self.usize_bn = nn.BatchNorm2d(self.out_channels)
        else :
            self.dsize_bn = nn.Identity()
            self.usize_bn = nn.Identity()
        self.upsample = nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=False)
        self.recovered_map = None
            
    def forward(self, x) :
        # Init recovered map
        self.recovered_map = None
        
        # Calculate boundary map
        out = self.conv(x)
        out = F.relu(self.dsize_bn(out))
        with torch.no_grad() :
            dense = F.relu(F.conv2d(input = x, weight = self.conv.weight, bias = self.conv.bias, stride = 1, padding = self.padding, dilation = self.dilation, groups = self.groups))
        dense = self.usize_bn(dense)
        sparse = self.upsample(out)
        self.recovered_map = torch.abs((sparse-dense))
        return out

