import torch


def softmax(xs: torch.Tensor, dim: int):
    from torch.nn.functional import softmax

    return softmax(xs, dim)
