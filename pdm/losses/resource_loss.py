import torch
import torch.nn as nn


class ResourceLoss(nn.Module):
    def __init__(self, p=0.9, loss_type='log'):
        super().__init__()
        assert loss_type in ["log", "mae", "mse"], f"Unknown loss type {loss_type}"
        self.p = p
        self.loss_type = loss_type

    def forward(self, resource_ratio):
        if self.loss_type == "log":
            if resource_ratio > self.p:
                resource_loss = torch.log(resource_ratio / self.p)
            else:
                resource_loss = torch.log(self.p / resource_ratio)
        elif self.loss_type == "mae":
            resource_loss = torch.abs(resource_ratio - self.p)
        else:
            resource_loss = (resource_ratio - self.p) ** 2

        return resource_loss
