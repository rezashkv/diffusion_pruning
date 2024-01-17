"""
    credit to:
    https://github.com/Alii-Ganjj/InterpretationsSteeredPruning/blob/96e5be3c721714bda76a4ab814ff5fe0ddf4417d/Models/hypernet.py (thanks!)
    """

from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config


class SimpleGate(nn.Module):
    def __init__(self, width):
        super(SimpleGate, self).__init__()
        self.weight = nn.Parameter(torch.randn(width))

    def forward(self):
        return self.weight


class HyperStructure(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, structure, input_dim=768, wn_flag=True, linear_bias=False):
        super(HyperStructure, self).__init__()

        self.structure = structure
        self.input_dim = input_dim
        self.linear_bias = linear_bias
        self.wn_flag = wn_flag

        self.width_list = [w for sub_width_list in self.structure['width'] for w in sub_width_list]
        self.depth_list = [d for sub_depth_list in self.structure['depth'] for d in sub_depth_list]

        width_linear_list = [nn.Linear(self.input_dim, self.width_list[i], bias=linear_bias) for i in
                             range(len(self.width_list))]
        depth_linear_layer = nn.Linear(self.input_dim, sum(self.depth_list), bias=linear_bias)

        linear_list = width_linear_list + [depth_linear_layer]

        if wn_flag:
            linear_list = [weight_norm(linear) for linear in linear_list]

        self.mh_fc = torch.nn.ModuleList(linear_list)
        self.initialize_weights()
        self.iteration = 0

    def forward(self, x):
        self.iteration += 1
        out = self._forward(x)
        return out

    def initialize_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def _forward(self, x):
        # x: B * L * D
        if self.mh_fc[0].weight.is_cuda:
            x = x.cuda()
        outputs = [self.mh_fc[i](x) for i in range(len(self.mh_fc))]
        out = torch.cat(outputs, dim=1)

        return out

    def print_param_stats(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                print(f"{name}: {param.mean()}, {param.std()}")

    def transform_structure_vector(self, inputs):
        assert inputs.shape[1] == (sum(self.width_list) + sum(self.depth_list))
        width_list = []
        depth_list = []
        width_vectors = inputs[:, :sum(self.width_list)]
        depth_vectors = inputs[:, sum(self.width_list):]
        start = 0
        for i in range(len(self.width_list)):
            end = start + self.width_list[i]
            width_list.append(width_vectors[:, start:end])
            start = end

        for i in range(sum(self.depth_list)):
            depth_list.append(depth_vectors[:, i])

        return {"width": width_list, "depth": depth_list}
