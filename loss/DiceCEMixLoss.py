import torch.nn as nn
import torch

from loss.SimpleDiceLoss import SimpleDiceLoss

class DiceCEMixLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceCEMixLoss, self).__init__()

    def forward(self, inputs, targets):

        dice_loss_obj = SimpleDiceLoss()
        ce_loss_obj = nn.CrossEntropyLoss()

        dice_loss = dice_loss_obj(inputs, targets)
        ce_loss = ce_loss_obj(inputs, targets)

        loss = 0.5 * ce_loss + 0.5 * dice_loss

        return loss