import torch
import torch.nn.functional as F


def sample_gumbel(shape, eps=1e-20, fixed_seed=False):
    if fixed_seed:
        u = torch.rand(shape, generator=torch.Generator().manual_seed(0))
    else:
        u = torch.rand(shape)
    return -torch.log(-torch.log(u + eps) + eps)


def vector_gumbel_softmax(logits, temperature, offset=0, force_width_non_zero=False, fixed_seed=False):
    gumbel_sample = sample_gumbel(logits.size(), fixed_seed=fixed_seed)
    if logits.is_cuda:
        gumbel_sample = gumbel_sample.cuda()

    y = logits + gumbel_sample + offset
    y_out = F.sigmoid(y / temperature)
    if not force_width_non_zero:
        return y_out
    else:
        y_out_h = hard_concrete(y_out).sum(dim=1)
        if (y_out_h > 0).all():
            return y_out
        else:
            y_out_h = hard_concrete(y_out).sum(dim=1)
            ind = (y_out_h == 0)
            new_y_out = y_out.clone()
            new_y_out[ind, 0] = y_out[ind, 0] + 0.5
            return new_y_out


def gumbel_softmax_sample(logits, temperature, offset=0, force_width_non_zero=False, fixed_seed=False):
    if not force_width_non_zero:
        gumbel_sample = sample_gumbel(logits.size(), fixed_seed=fixed_seed)
        if logits.is_cuda:
            gumbel_sample = gumbel_sample.cuda()

        y = logits + gumbel_sample + offset
        return F.sigmoid(y / temperature)

    else:
        y_out = vector_gumbel_softmax(logits=logits, temperature=temperature, offset=offset, force_width_non_zero=True,
                                      fixed_seed=fixed_seed)
        return y_out


def importance_gumbel_softmax_sample(logits, temperature, offset=0, fixed_seed=False):
    x = torch.softmax(logits, dim=1)
    x = torch.cumsum(x, dim=1)
    x = torch.flip(x, dims=[1])

    eps = 1e-6
    # inverse sigmoid function. add eps to avoid numerical instability.
    x = torch.log(x + eps) - torch.log1p(-(x - eps))

    gumbel_sample = sample_gumbel(x.size(), fixed_seed=fixed_seed)
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
