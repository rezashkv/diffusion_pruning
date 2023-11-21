"""
This script contains the implementations of gate functions and their gradient calculation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.function


class soft_gate(nn.Module):
    def __init__(self, width, base_width=-1, concrete_flag=False, margin=0):
        super(soft_gate, self).__init__()
        if base_width == -1:
            base_width = width
        self.weights = nn.Parameter(torch.ones(width))
        self.training_flag = True

        self.concrete_flag = concrete_flag

        self.g_w = torch.Tensor([float(base_width) / float(width)])

        self.margin = margin
        if concrete_flag:
            self.margin = 0

    def forward(self, input):
        if not self.training_flag:
            return input
        self.weights.data.copy_(self.clip_value(self.weights.data))
        if len(input.size()) == 2:

            if self.concrete_flag:
                gate_f = custom_STE.apply(self.weights, False)
            else:
                gate_f = custom_STE.apply(self.weights, self.training, self.g_w)

            gate_f = gate_f.unsqueeze(0)

            if input.is_cuda:
                gate_f = gate_f.cuda()

            input = gate_f.expand_as(input) * input
            return input

        elif len(input.size()) == 4:

            if self.concrete_flag:
                gate_f = custom_STE.apply(self.weights, False)
            else:
                gate_f = custom_STE.apply(self.weights, self.training, self.g_w)
            gate_f = gate_f.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            if input.is_cuda:
                gate_f = gate_f.cuda()
            input = gate_f.expand_as(input) * input

            return input

    def clip_value(self, x):
        x[x > 1 - self.margin] = 1 - self.margin
        x[x < 0 + self.margin] = self.margin
        return x


class VirtualGate(nn.Module):
    def __init__(self, width, bs=1):
        super(VirtualGate, self).__init__()
        self.g_w = 1
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

        elif len(x.size()) == 4:
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
        if x.is_cuda:
            gate_f = gate_f.cuda()
        for _ in range(len(x.size()) - 2):
            gate_f = gate_f.unsqueeze(-1)
        gate_f = gate_f.expand_as(x)
        x = gate_f * x
        return x


def tanh_gradient(x, T=4, b=0.5):
    value_pos = torch.exp(T * (x - b))
    value_neg = torch.exp(-T * (x - b))
    return 2 * T / (value_pos + value_neg)


class AC_layer(nn.Module):
    def __init__(self, num_class=10):
        super(AC_layer, self).__init__()
        self.fc = nn.Linear(num_class, num_class)
        self.num_class = num_class

    def forward(self, input):
        b_size, n_c, w, h = input.size()
        input = input.view(b_size, 1, -1)
        input = F.adaptive_avg_pool1d(input, self.num_class)
        out = self.fc(input.squeeze())
        return out


class custom_STE(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, train, grad_w=1):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        train = train
        ctx.grad_w = grad_w
        if train is True:
            ctx.save_for_backward(input)
            input_clone = input.clone()
            input_clone = prob_round_torch(input_clone)
        else:
            ctx.save_for_backward(input)
            input_clone = input.clone()
            input_clone[input >= 0.5] = 1
            input_clone[input < 0.5] = 0

        return input_clone.float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input > 1] = 0
        grad_input[input < 0] = 0
        gw = ctx.grad_w
        if grad_input.is_cuda and type(gw) is not int:
            gw = gw.cuda()

        return grad_input * gw, None, None


class custom_grad_weight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grad_w=1):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.grad_w = grad_w
        input_clone = input.clone()
        return input_clone.float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        grad_input = grad_output.clone()
        gw = ctx.grad_w
        if grad_input.is_cuda and type(gw) is not int:
            gw = gw.cuda()

        return grad_input * gw, None, None


def prob_round_torch(x):
    if x.is_cuda:
        stochastic_round = torch.rand(x.size(0)).cuda() < x
    else:
        stochastic_round = torch.rand(x.size(0)) < x
    return stochastic_round
