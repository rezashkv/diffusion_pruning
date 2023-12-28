import torch
import torch.nn.functional as F


def sample_gumbel(shape, eps=1e-20):
    u = torch.rand(shape)
    return -torch.log(-torch.log(u + eps) + eps)


def gumbel_softmax_sample(logits, temperature, offset=0):
    gumbel_sample = sample_gumbel(logits.size())
    if logits.is_cuda:
        gumbel_sample = gumbel_sample.cuda()

    y = logits + gumbel_sample + offset
    return F.sigmoid(y / temperature)


def importance_gumble_softmax_sample(logits, temperature, offset=0):
    x = torch.softmax(logits, dim=1)
    x = torch.cumsum(x, dim=1)
    x = torch.flip(x, dims=[1])

    eps = 1e-8
    # inverse sigmoid function. add eps to avoid numerical instability.
    x = torch.log(x - eps) - torch.log1p(-x)

    gumbel_sample = sample_gumbel(x.size())
    if logits.is_cuda:
        gumbel_sample = gumbel_sample.cuda()
        x = x.cuda()

    y = x + gumbel_sample + offset
    return F.sigmoid(y / temperature)


def hard_concrete(out):
    out_hard = torch.zeros(out.size())
    out_hard[out >= 0.5] = 1
    out_hard[out < 0.5] = 0
    if out.is_cuda:
        out_hard = out_hard.cuda()
    # Straight through estimation
    out_hard = (out_hard - out).detach() + out
    return out_hard
