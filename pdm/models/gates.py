"""
This script contains the implementations of gate functions and their gradient calculation.
"""
import torch
import torch.nn as nn
import torch.autograd.function


class VirtualGate(nn.Module):
    def __init__(self, width, bs=1):
        super(VirtualGate, self).__init__()
        self.width = width
        self.gate_f = torch.ones(bs, width)

    def forward(self, x, dim=1):
        mask = self.gate_f.repeat_interleave(x.shape[dim] // self.width, dim=1)
        if dim == -1:
            expand_dim = -2
        else:
            expand_dim = -1
        for _ in range(len(x.shape) - len(mask.shape)):
            mask = mask.unsqueeze(expand_dim)

        # to handle cfg where actual batch size is double the value of the batch size used to create the gate.
        if mask.shape[0] != x.shape[0]:
            mask = mask.repeat(x.shape[0] // mask.shape[0], 1, 1, 1)
        x = mask.expand_as(x) * x
        return x

    def set_structure_value(self, value):
        self.gate_f = value


class WidthGate(VirtualGate):
    def __init__(self, width):
        super(WidthGate, self).__init__(width)


class DepthGate(VirtualGate):
    def __init__(self, width):
        super(DepthGate, self).__init__(width)

    def forward(self, x):
        input_hidden_states, output_tensor = x
        mask = self.gate_f.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        if mask.shape[0] != output_tensor.shape[0]:
            mask = mask.repeat(output_tensor.shape[0] // mask.shape[0], 1, 1, 1)
        output = (1 - mask) * input_hidden_states + mask * output_tensor
        return output


class LinearWidthGate(WidthGate):
    def __init__(self, width):
        super(LinearWidthGate, self).__init__(width)

    def forward(self, x):
        mask = self.gate_f.repeat_interleave(x.shape[-1] // self.width, dim=1).unsqueeze(1)
        # to handle cfg where actual batch size is double the value of the batch size used to create the gate.
        if mask.shape[0] != x.shape[0]:
            mask = mask.repeat(x.shape[0] // mask.shape[0], 1, 1)
        x = mask.expand_as(x) * x
        return x
