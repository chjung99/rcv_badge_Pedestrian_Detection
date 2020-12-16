from torchcv.models.ssd.multispectral_net import MSSDPed
from torchcv.models.ssd.net import SSDPed
# from torchcv.models.ssd.net import SSD300, SSD512
from torchcv.models.box_coder import SSDBoxCoder

import torch
import torch.nn as nn
import torch.nn.init as init

def weights_init(m):
    if isinstance(m, nn.Conv2d):        
        init.normal_(m.weight.data, std=0.01)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

