import torch
import torch.nn as nn


class EPE3DLoss(nn.Module):
    def __init__(self):
        super(EPE3DLoss, self).__init__()

    def forward(self, input, target):
        return torch.norm(input - target, p=2, dim=1)