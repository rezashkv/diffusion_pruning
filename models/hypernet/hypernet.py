"""
    credit to:
    https://github.com/Alii-Ganjj/InterpretationsSteeredPruning/blob/96e5be3c721714bda76a4ab814ff5fe0ddf4417d/Models/hypernet.py (thanks!)
    """

from __future__ import absolute_import

import os
from typing import Union, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import ModelMixin

from torch.nn.utils import weight_norm


class SimpleGate(nn.Module):
    def __init__(self, width):
        super(SimpleGate, self).__init__()
        self.weight = nn.Parameter(torch.randn(width))

    def forward(self):
        return self.weight


class HyperStructure(ModelMixin):
    def __init__(self, input_dim=1024, seq_len=77, structure=None, T=0.4, sparsity=0, base=2, wn_flag=True):
        super(HyperStructure, self).__init__()

        self.T = T
        self.structure = structure
        self.input_dim = input_dim

        self.Bi_GRU = nn.GRU(self.input_dim, self.input_dim, bidirectional=True)

        self.bn1 = nn.LayerNorm([2 * self.input_dim])

        self.h0 = torch.zeros(2, seq_len, self.input_dim)

        self.sparsity = [sparsity] * len(structure)

        if wn_flag:
            self.linear_list = [weight_norm(nn.Linear(2 * self.input_dim, structure[i], bias=False))
                                for i in range(len(structure))]
        else:
            self.linear_list = [nn.Linear(2 * self.input_dim, structure[i], bias=False, ) for i
                                in range(len(structure))]

        self.mh_fc = torch.nn.ModuleList(self.linear_list)
        self.base = base

        self.iteration = 0

        # print the mean and std of the weights of the gru

    def forward(self, x):
        # expand x so the first dim has len(structure) copies
        # x = x.expand(len(self.structure), x.size(1), x.size(2))
        self.iteration += 1
        out = self._forward(x)
        # if not self.training:
        #     out = hard_concrete(out)
        return out

    # def transform_output(self, x):
    #     arch_vector = []
    #     start = 0
    #     for i in range(len(self.structure)):
    #         end = start + self.structure[i]
    #         arch_vector.append(x[start:end])
    #         start = end
    #
    #     return arch_vector

    # def resource_output(self, x):
    #     out = self._forward(x)
    #     out = hard_concrete(out)
    #     return out.squeeze()

    def _forward(self, x):
        if self.bn1.weight.is_cuda:
            x = x.cuda()
            self.h0 = self.h0.cuda()
        outputs, hn = self.Bi_GRU(x, self.h0)
        outputs = outputs.sum(dim=1).unsqueeze(1).expand(outputs.size(0), len(self.structure), outputs.size(2))
        outputs = [F.relu(self.bn1(outputs[:, i, :])) for i in range(len(self.structure))]
        outputs = [self.mh_fc[i](outputs[i]) for i in range(len(self.mh_fc))]

        out = torch.cat(outputs, dim=1)
        # out = gumbel_softmax_sample(out, temperature=self.T, offset=self.base)

        return out

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        save_function: Callable = None,
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        push_to_hub: bool = False,
        **kwargs,
    ):
        # save the hyper_net state
        torch.save(self.state_dict(), os.path.join(save_directory, "hyper_net.pt"))

    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        # load the hyper_net state
        model = cls(**kwargs)
        model.load_state_dict(torch.load(os.path.join(pretrained_model_name_or_path, "hyper_net.pt")))
        return model

    def print_param_stats(self):
        print("HyperNet")
        for name, param in self.named_parameters():
            if "weight" in name:
                print(f"{name}: {param.mean()}, {param.std()}")
        print("")
