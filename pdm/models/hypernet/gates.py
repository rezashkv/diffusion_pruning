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

    def forward(self, x):
        if len(x.size()) == 2:
            gate_f = self.gate_f
            if x.is_cuda:
                gate_f = gate_f.cuda()
            # gate_f has width equal to number of groups, so we need to expand it to match the input size
            x = gate_f.expand_as(x) * x
            return x

        elif len(x.size()) == 3:
            gate_f = self.gate_f.unsqueeze(-1).unsqueeze(-1)
            if x.is_cuda:
                gate_f = gate_f.cuda()
            x = gate_f.expand_as(x) * x
            return x

    def set_structure_value(self, value):
        self.gate_f = value


class BlockVirtualGate(VirtualGate):
    def __init__(self, width):
        super(BlockVirtualGate, self).__init__(width)

    def forward(self, x):
        gate_f = torch.repeat_interleave(self.gate_f, x.shape[1] // self.width, dim=1)
        if gate_f.shape[0] > 1 and gate_f.shape[0] != x.shape[0]:
            # cat the gate_f to itself to match the batch size for classifier-free guidance
            gate_f = torch.cat([gate_f] * (x.shape[0] // gate_f.shape[0]))
        if x.is_cuda:
            gate_f = gate_f.cuda()
        for _ in range(len(x.size()) - 2):
            gate_f = gate_f.unsqueeze(-1)
        gate_f = gate_f.expand_as(x)
        x = gate_f * x
        return x


class LinearVirtualGate(VirtualGate):
    def __init__(self, width):
        super(LinearVirtualGate, self).__init__(width)

    def forward(self, x):
        gate_f = self.gate_f
        for _ in range(len(x.size()) - 2):
            gate_f = gate_f.unsqueeze(1)
        if x.is_cuda:
            gate_f = gate_f.cuda()
        x = gate_f.expand_as(x) * x
        return x
