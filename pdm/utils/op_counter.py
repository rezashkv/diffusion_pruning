'''
This opcounter is adapted from https://github.com/sovrasov/macs-counter.pytorch

Copyright (C) 2021 Sovrasov V. - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
'''

import numpy as np
import torch

from diffusers.models.attention_processor import SpatialNorm
from diffusers.models.normalization import AdaGroupNorm
# from diffusers.models.activations import GEGLU
from diffusers.models.lora import (LoRACompatibleConv, LoRACompatibleLinear)
from pdm.models.unet.blocks import GatedAttention
from diffusers.models.attention_processor import Attention
import sys
from functools import partial
import torch.nn as nn


@torch.no_grad()
def count_ops_and_params(model, example_inputs):
    global CUSTOM_MODULES_MAPPING
    macs_model = add_macs_counting_methods(model)
    macs_model.eval()
    macs_model.start_macs_count(ost=sys.stdout, verbose=False,
                                  ignore_list=[])
    if isinstance(example_inputs, dict):
        _ = macs_model(**example_inputs)
    elif isinstance(example_inputs, (tuple, list)):
        _ = macs_model(*example_inputs)
    else:
        _ = macs_model(example_inputs)
    macs_count, params_count = macs_model.compute_average_macs_cost()
    macs_model.stop_macs_count()
    CUSTOM_MODULES_MAPPING = {}
    return macs_count, params_count


def empty_macs_counter_hook(module, input, output):
    module.__macs__ += 0


def upsample_macs_counter_hook(module, input, output):
    output_size = output[0]
    batch_size = output_size.shape[0]
    output_elements_count = batch_size
    for val in output_size.shape[1:]:
        output_elements_count *= val
    module.__macs__ += int(output_elements_count)


def relu_macs_counter_hook(module, input, output):
    active_elements_count = output.numel()
    module.__macs__ += int(active_elements_count)


def silu_macs_counter_hook(module, input, output):
    active_elements_count = output.numel()
    module.__macs__ += int(active_elements_count * 2)


def linear_macs_counter_hook(module, input, output):
    input = input[0]
    # pytorch checks dimensions, so here we don't care much
    output_last_dim = output.shape[-1]
    bias_macs = output_last_dim if module.bias is not None else 0
    module.__macs__ += int(np.prod(input.shape) * output_last_dim + bias_macs)


def pool_macs_counter_hook(module, input, output):
    input = input[0]
    module.__macs__ += int(np.prod(input.shape))


def bn_macs_counter_hook(module, input, output):
    input = input[0]

    batch_macs = np.prod(input.shape)
    if module.affine:
        batch_macs *= 2
    module.__macs__ += int(batch_macs)


def layer_norm_macs_counter_hook(module, input, output):
    input = input[0]

    batch_macs = np.prod(input.shape)
    module.__macs__ += int(batch_macs)


def conv_macs_counter_hook(conv_module, input, output):
    # Can have multiple inputs, getting the first one
    input = input[0]

    batch_size = input.shape[0]
    output_dims = list(output.shape[2:])

    kernel_dims = list(conv_module.kernel_size)
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_macs = int(np.prod(kernel_dims)) * \
                              in_channels * filters_per_channel

    active_elements_count = batch_size * int(np.prod(output_dims))

    overall_conv_macs = conv_per_position_macs * active_elements_count

    bias_macs = 0

    if conv_module.bias is not None:
        bias_macs = out_channels * active_elements_count

    overall_macs = overall_conv_macs + bias_macs

    conv_module.__macs__ += int(overall_macs)


def rnn_macs(macs, rnn_module, w_ih, w_hh, input_size):
    # matrix matrix mult ih state and internal state
    macs += w_ih.shape[0] * w_ih.shape[1]
    # matrix matrix mult hh state and internal state
    macs += w_hh.shape[0] * w_hh.shape[1]
    if isinstance(rnn_module, (nn.RNN, nn.RNNCell)):
        # add both operations
        macs += rnn_module.hidden_size
    elif isinstance(rnn_module, (nn.GRU, nn.GRUCell)):
        # hadamard of r
        macs += rnn_module.hidden_size
        # adding operations from both states
        macs += rnn_module.hidden_size * 3
        # last two hadamard product and add
        macs += rnn_module.hidden_size * 3
    elif isinstance(rnn_module, (nn.LSTM, nn.LSTMCell)):
        # adding operations from both states
        macs += rnn_module.hidden_size * 4
        # two hadamard product and add for C state
        macs += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
        # final hadamard
        macs += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
    return macs


def rnn_macs_counter_hook(rnn_module, input, output):
    """
    Takes into account batch goes at first position, contrary
    to pytorch common rule (but actually it doesn't matter).
    If sigmoid and tanh are hard, only a comparison macs should be accurate
    """
    macs = 0
    # input is a tuple containing a sequence to process and (optionally) hidden state
    inp = input[0]
    batch_size = inp[0].shape[0]
    seq_length = inp[0].shape[1]
    num_layers = rnn_module.num_layers

    for i in range(num_layers):
        w_ih = rnn_module.__getattr__('weight_ih_l' + str(i))
        w_hh = rnn_module.__getattr__('weight_hh_l' + str(i))
        if i == 0:
            input_size = rnn_module.input_size
        else:
            input_size = rnn_module.hidden_size
        macs = rnn_macs(macs, rnn_module, w_ih, w_hh, input_size)
        if rnn_module.bias:
            b_ih = rnn_module.__getattr__('bias_ih_l' + str(i))
            b_hh = rnn_module.__getattr__('bias_hh_l' + str(i))
            macs += b_ih.shape[0] + b_hh.shape[0]

    macs *= batch_size
    macs *= seq_length
    if rnn_module.bidirectional:
        macs *= 2
    rnn_module.__macs__ += int(macs)


def rnn_cell_macs_counter_hook(rnn_cell_module, input, output):
    macs = 0
    inp = input[0]
    batch_size = inp.shape[0]
    w_ih = rnn_cell_module.__getattr__('weight_ih')
    w_hh = rnn_cell_module.__getattr__('weight_hh')
    input_size = inp.shape[1]
    macs = rnn_macs(macs, rnn_cell_module, w_ih, w_hh, input_size)
    if rnn_cell_module.bias:
        b_ih = rnn_cell_module.__getattr__('bias_ih')
        b_hh = rnn_cell_module.__getattr__('bias_hh')
        macs += b_ih.shape[0] + b_hh.shape[0]

    macs *= batch_size
    rnn_cell_module.__macs__ += int(macs)


def multihead_attention_counter_hook(multihead_attention_module, input, output):
    macs = 0
    q, k, v = input

    batch_first = multihead_attention_module.batch_first \
        if hasattr(multihead_attention_module, 'batch_first') else False
    if batch_first:
        batch_size = q.shape[0]
        len_idx = 1
    else:
        batch_size = q.shape[1]
        len_idx = 0

    dim_idx = 2

    qdim = q.shape[dim_idx]
    kdim = k.shape[dim_idx]
    vdim = v.shape[dim_idx]

    qlen = q.shape[len_idx]
    klen = k.shape[len_idx]
    vlen = v.shape[len_idx]

    num_heads = multihead_attention_module.num_heads
    assert qdim == multihead_attention_module.embed_dim

    if multihead_attention_module.kdim is None:
        assert kdim == qdim
    if multihead_attention_module.vdim is None:
        assert vdim == qdim

    macs = 0

    # Q scaling
    macs += qlen * qdim

    # Initial projections
    macs += (
            (qlen * qdim * qdim)  # QW
            + (klen * kdim * kdim)  # KW
            + (vlen * vdim * vdim)  # VW
    )

    if multihead_attention_module.in_proj_bias is not None:
        macs += (qlen + klen + vlen) * qdim

    # attention heads: scale, matmul, softmax, matmul
    qk_head_dim = qdim // num_heads
    v_head_dim = vdim // num_heads

    head_macs = (
            (qlen * klen * qk_head_dim)  # QK^T
            + (qlen * klen)  # softmax
            + (qlen * klen * v_head_dim)  # AV
    )

    macs += num_heads * head_macs

    # final projection, bias is always enabled
    macs += qlen * vdim * (vdim + 1)

    macs *= batch_size
    multihead_attention_module.__macs__ += int(macs)


def gated_attention_counter_hook(gated_attention_module, input, output):
    if gated_attention_module.total_macs == 0. or gated_attention_module.pruned:
        gated_attention_module.total_macs, gated_attention_module.prunable_macs = 0., 0.
        # SpatialNorm
        if gated_attention_module.spatial_norm is not None:
            gated_attention_module.total_macs += gated_attention_module.spatial_norm.__macs__

        # GroupNorm
        if gated_attention_module.group_norm is not None:
            gated_attention_module.total_macs += gated_attention_module.group_norm.__macs__

        # NormCross
        if gated_attention_module.norm_cross:
            gated_attention_module.total_macs += gated_attention_module.norm_cross.__macs__

        # to_q
        gated_attention_module.total_macs += gated_attention_module.to_q.__macs__
        gated_attention_module.prunable_macs += gated_attention_module.to_q.__macs__

        # to_k
        gated_attention_module.total_macs += gated_attention_module.to_k.__macs__
        gated_attention_module.prunable_macs += gated_attention_module.to_k.__macs__

        # to_v
        gated_attention_module.total_macs += gated_attention_module.to_v.__macs__
        gated_attention_module.prunable_macs += gated_attention_module.to_v.__macs__

        attn_macs = 0
        batch_size, seq_len, _ = output.shape
        dim = gated_attention_module.to_q.out_features
        num_heads, head_dim = gated_attention_module.heads, dim // gated_attention_module.heads

        head_macs = (
                (seq_len * seq_len * head_dim)  # QK^T
                + (seq_len * seq_len)  # softmax
                + (seq_len * seq_len * head_dim)  # AV
        )

        attn_macs += num_heads * head_macs

        gated_attention_module.total_macs += attn_macs
        gated_attention_module.prunable_macs += attn_macs

        # to_out
        gated_attention_module.total_macs += gated_attention_module.to_out[0].__macs__
        gated_attention_module.prunable_macs += gated_attention_module.to_out[0].__macs__

    gated_attention_module.__macs__ = gated_attention_module.total_macs


def attention_counter_hook(attention_module, input, output):
    total_macs = 0
    if attention_module.spatial_norm is not None:
        total_macs += attention_module.spatial_norm.__macs__

    # GroupNorm
    if attention_module.group_norm is not None:
        total_macs += attention_module.group_norm.__macs__

    # NormCross
    if attention_module.norm_cross:
        total_macs += attention_module.norm_cross.__macs__

    # to_q
    total_macs += attention_module.to_q.__macs__

    # to_k
    total_macs += attention_module.to_k.__macs__

    # to_v
    total_macs += attention_module.to_v.__macs__

    attn_macs = 0
    batch_size, seq_len, _ = output.shape
    dim = attention_module.to_q.out_features
    num_heads, head_dim = attention_module.heads, dim // attention_module.heads

    head_macs = (
            (seq_len * seq_len * head_dim)  # QK^T
            + (seq_len * seq_len)  # softmax
            + (seq_len * seq_len * head_dim)  # AV
    )

    attn_macs += num_heads * head_macs

    total_macs += attn_macs

    # to_out
    total_macs += attention_module.to_out[0].__macs__

    attention_module.__macs__ = total_macs


CUSTOM_MODULES_MAPPING = {}

MODULES_MAPPING = {
    # convolutions
    nn.Conv1d: conv_macs_counter_hook,
    nn.Conv2d: conv_macs_counter_hook,
    nn.Conv3d: conv_macs_counter_hook,
    LoRACompatibleConv: conv_macs_counter_hook,

    # activations
    nn.ReLU: relu_macs_counter_hook,
    nn.PReLU: relu_macs_counter_hook,
    nn.ELU: relu_macs_counter_hook,
    nn.LeakyReLU: relu_macs_counter_hook,
    nn.ReLU6: relu_macs_counter_hook,
    nn.SiLU: silu_macs_counter_hook,

    # poolings
    nn.MaxPool1d: pool_macs_counter_hook,
    nn.AvgPool1d: pool_macs_counter_hook,
    nn.AvgPool2d: pool_macs_counter_hook,
    nn.MaxPool2d: pool_macs_counter_hook,
    nn.MaxPool3d: pool_macs_counter_hook,
    nn.AvgPool3d: pool_macs_counter_hook,
    nn.AdaptiveMaxPool1d: pool_macs_counter_hook,
    nn.AdaptiveAvgPool1d: pool_macs_counter_hook,
    nn.AdaptiveMaxPool2d: pool_macs_counter_hook,
    nn.AdaptiveAvgPool2d: pool_macs_counter_hook,
    nn.AdaptiveMaxPool3d: pool_macs_counter_hook,
    nn.AdaptiveAvgPool3d: pool_macs_counter_hook,

    # BNs
    nn.BatchNorm1d: bn_macs_counter_hook,
    nn.BatchNorm2d: bn_macs_counter_hook,
    nn.BatchNorm3d: bn_macs_counter_hook,

    nn.InstanceNorm1d: bn_macs_counter_hook,
    nn.InstanceNorm2d: bn_macs_counter_hook,
    nn.InstanceNorm3d: bn_macs_counter_hook,
    nn.GroupNorm: bn_macs_counter_hook,
    SpatialNorm: bn_macs_counter_hook,
    AdaGroupNorm: bn_macs_counter_hook,
    nn.LayerNorm: layer_norm_macs_counter_hook,

    # FC
    nn.Linear: linear_macs_counter_hook,
    LoRACompatibleLinear: linear_macs_counter_hook,

    # Upscale
    nn.Upsample: upsample_macs_counter_hook,

    # Deconvolution
    nn.ConvTranspose1d: conv_macs_counter_hook,
    nn.ConvTranspose2d: conv_macs_counter_hook,
    nn.ConvTranspose3d: conv_macs_counter_hook,

    # RNN
    nn.RNN: rnn_macs_counter_hook,
    nn.GRU: rnn_macs_counter_hook,
    nn.LSTM: rnn_macs_counter_hook,
    nn.RNNCell: rnn_cell_macs_counter_hook,
    nn.LSTMCell: rnn_cell_macs_counter_hook,
    nn.GRUCell: rnn_cell_macs_counter_hook,
    nn.MultiheadAttention: multihead_attention_counter_hook,

    Attention: attention_counter_hook,
    GatedAttention: gated_attention_counter_hook,
}

if hasattr(nn, 'GELU'):
    MODULES_MAPPING[nn.GELU] = relu_macs_counter_hook


def accumulate_macs(self):
    if is_supported_instance(self):
        return self.__macs__
    else:
        sum = 0
        for m in self.children():
            sum += m.accumulate_macs()
        return sum


def get_model_parameters_number(model):
    # params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_num = sum(p.numel() for p in model.parameters())
    return params_num


def add_macs_counting_methods(net_main_module):
    # adding additional methods to the existing module object,
    # this is done this way so that each function has access to self object
    net_main_module.start_macs_count = start_macs_count.__get__(net_main_module)
    net_main_module.stop_macs_count = stop_macs_count.__get__(net_main_module)
    net_main_module.reset_macs_count = reset_macs_count.__get__(net_main_module)
    net_main_module.compute_average_macs_cost = compute_average_macs_cost.__get__(
        net_main_module)

    net_main_module.reset_macs_count()

    return net_main_module


def compute_average_macs_cost(self):
    """
    A method that will be available after add_macs_counting_methods() is called
    on a desired net object.
    Returns current mean macs consumption per image.
    """

    for m in self.modules():
        m.accumulate_macs = accumulate_macs.__get__(m)

    macs_sum = self.accumulate_macs()

    for m in self.modules():
        if hasattr(m, 'accumulate_macs'):
            del m.accumulate_macs

    params_sum = get_model_parameters_number(self)
    return macs_sum / self.__batch_counter__, params_sum


def start_macs_count(self, **kwargs):
    """
    A method that will be available after add_macs_counting_methods() is called
    on a desired net object.
    Activates the computation of mean macs consumption per image.
    Call it before you run the network.
    """
    add_batch_counter_hook_function(self)

    seen_types = set()

    def add_macs_counter_hook_function(module, ost, verbose, ignore_list):
        if type(module) in ignore_list:
            seen_types.add(type(module))
            if is_supported_instance(module):
                module.__params__ = 0
        elif is_supported_instance(module):
            if hasattr(module, '__macs_handle__'):
                return
            if type(module) in CUSTOM_MODULES_MAPPING:
                handle = module.register_forward_hook(
                    CUSTOM_MODULES_MAPPING[type(module)])
            else:
                handle = module.register_forward_hook(MODULES_MAPPING[type(module)])
            module.__macs_handle__ = handle
            seen_types.add(type(module))
        else:
            if verbose and not type(module) in (nn.Sequential, nn.ModuleList) and \
                    not type(module) in seen_types:
                print('Warning: module ' + type(module).__name__ +
                      ' is treated as a zero-op.', file=ost)
            seen_types.add(type(module))

    self.apply(partial(add_macs_counter_hook_function, **kwargs))


def stop_macs_count(self):
    """
    A method that will be available after add_macs_counting_methods() is called
    on a desired net object.
    Stops computing the mean macs consumption per image.
    Call whenever you want to pause the computation.
    """
    remove_batch_counter_hook_function(self)
    self.apply(remove_macs_counter_hook_function)
    # self.apply(remove_macs_counter_variables)


def reset_macs_count(self):
    """
    A method that will be available after add_macs_counting_methods() is called
    on a desired net object.
    Resets statistics computed so far.
    """
    add_batch_counter_variables_or_reset(self)
    self.apply(add_macs_counter_variable_or_reset)


# ---- Internal functions
def batch_counter_hook(module, input, output):
    batch_size = 1
    if len(input) > 0:
        # Can have multiple inputs, getting the first one
        input = input[0]
        batch_size = len(input)
    else:
        pass
        print('\nWarning! No positional inputs found for a module,'
              ' assuming batch size is 1.')
    module.__batch_counter__ += batch_size


def add_batch_counter_variables_or_reset(module):
    module.__batch_counter__ = 0


def add_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        return

    handle = module.register_forward_hook(batch_counter_hook)
    module.__batch_counter_handle__ = handle


def remove_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        module.__batch_counter_handle__.remove()
        del module.__batch_counter_handle__


def add_macs_counter_variable_or_reset(module):
    if is_supported_instance(module):
        if hasattr(module, '__macs__') or hasattr(module, '__params__'):
            # print('Warning: variables __macs__ or __params__ are already '
            #       'defined for the module' + type(module).__name__ +
            #       ' ptmacs can affect your code!')
            module.__ptmacs_backup_macs__ = module.__macs__
            module.__ptmacs_backup_params__ = module.__params__
        module.__macs__ = 0
        module.__params__ = get_model_parameters_number(module)


def is_supported_instance(module):
    if type(module) in MODULES_MAPPING or type(module) in CUSTOM_MODULES_MAPPING:
        return True
    return False


def remove_macs_counter_hook_function(module):
    if is_supported_instance(module):
        if hasattr(module, '__macs_handle__'):
            module.__macs_handle__.remove()
            del module.__macs_handle__


def remove_macs_counter_variables(module):
    if is_supported_instance(module):
        if hasattr(module, '__macs__'):
            del module.__macs__
            if hasattr(module, '__ptmacs_backup_macs__'):
                module.__macs__ = module.__ptmacs_backup_macs__
        if hasattr(module, '__params__'):
            del module.__params__
            if hasattr(module, '__ptmacs_backup_params__'):
                module.__params__ = module.__ptmacs_backup_params__
