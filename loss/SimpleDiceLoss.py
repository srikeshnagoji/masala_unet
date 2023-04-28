import torch.nn as nn
import torch

class SimpleDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SimpleDiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        #comment out if your model contains a sigmoid or equivalent activation layer
#         inputs = F.sigmoid(inputs)
        
        #flatten label and prediction tensors

        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # For SWIN..
        # inputs = inputs.contiguous().view(-1)
        # targets = targets.contiguous().view(-1)
        # print(inputs.size(), targets.size())

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        return 1 - dice