import math

import einops
import torch
from torch import nn

class MSR(nn.Module):
    def __init__(self, dim, in_resolution, num_heads):
        self.dim = dim
        self.in_resolution = in_resolution
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_g = nn.Linear(dim, dim)
        self.decay = 1 - torch.exp(-5 - num_heads)

    def _get_D(self, batch_size):
        n, m = self.in_resolution
        D = torch.ones(batch_size, n, m, self.dim)
        n = torch.arange(n).unsqueeze(1)
        m = torch.arange(m).unsqueeze(0)
        D = (self.decay ** abs(n - m)).float()

        return D

    def forward(self, x):
        pass

def _get_D(in_resolution, batch_size, dim, decay):
    n_, m_ = in_resolution
    D = torch.ones((batch_size, dim, n_, m_))
    n = torch.arange(n_).unsqueeze(1)
    m = torch.arange(m_).unsqueeze(0)
    D_ = (decay ** abs(n - m)).float().unsqueeze(0)
    D = D_.expand(batch_size, dim, n_, m_)
    print(D.shape)

    return D

if __name__ == "__main__":
    x = torch.randn(1, 3, 8, 8)
    B, C, H, W = x.shape

    D = _get_D(in_resolution=(H, W), batch_size=B, dim=C, decay=1 - pow(2, -5 - 8))
    print(D)