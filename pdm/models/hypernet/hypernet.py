"""
    credit to:
    https://github.com/Alii-Ganjj/InterpretationsSteeredPruning/blob/96e5be3c721714bda76a4ab814ff5fe0ddf4417d/Models/hypernet.py (thanks!)
    """

from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config

from torch.nn.utils.parametrizations import weight_norm


class SimpleGate(nn.Module):
    def __init__(self, width):
        super(SimpleGate, self).__init__()
        self.weight = nn.Parameter(torch.randn(width))

    def forward(self):
        return self.weight


class HyperStructure(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, structure, input_dim=1024, inner_dim=1024, seq_len=77, T=0.4, sparsity=0, base=2, wn_flag=True):
        super(HyperStructure, self).__init__()

        self.T = T
        self.structure = structure
        self.input_dim = input_dim

        self.width_list = [w for sub_width_list in self.structure['width'] for w in sub_width_list]
        self.depth_list = [d for sub_depth_list in self.structure['depth'] for d in sub_depth_list]

        # gru_hidden_dim = 2 * self.input_dim
        gru_hidden_dim = inner_dim

        self.Bi_GRU = nn.GRU(self.input_dim, gru_hidden_dim, bidirectional=True)

        self.bn1 = nn.LayerNorm([2 * gru_hidden_dim])

        self.h0 = torch.zeros(2, seq_len, gru_hidden_dim)

        self.sparsity = [sparsity] * len(structure)

        if wn_flag:
            self.linear_list = [weight_norm(nn.Linear(2 * gru_hidden_dim, self.width_list[i], bias=False))
                                for i in range(len(self.width_list))]
            self.linear_list.append(weight_norm(nn.Linear(2 * gru_hidden_dim, sum(self.depth_list), bias=False)))

        else:
            self.linear_list = [nn.Linear(2 * gru_hidden_dim, self.width_list[i], bias=False) for i
                                in range(len(self.width_list))]
            self.linear_list.append(nn.Linear(2 * gru_hidden_dim, sum(self.depth_list), bias=False))

        self.mh_fc = torch.nn.ModuleList(self.linear_list)
        self.base = base

        self.iteration = 0

    def forward(self, x):
        self.iteration += 1
        out = self._forward(x)
        return out

    def _forward(self, x):
        # x: B * L * D
        if self.bn1.weight.is_cuda:
            x = x.cuda()
            self.h0 = self.h0.cuda()
        outputs, hn = self.Bi_GRU(x, self.h0)
        outputs = outputs.sum(dim=1).unsqueeze(1).expand(outputs.size(0), len(self.structure), outputs.size(2))
        outputs = [F.relu(self.bn1(outputs[:, i, :])) for i in range(len(self.structure))]
        outputs = [self.mh_fc[i](outputs[i]) for i in range(len(self.mh_fc))]

        out = torch.cat(outputs, dim=1)

        return out

    def print_param_stats(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                print(f"{name}: {param.mean()}, {param.std()}")
