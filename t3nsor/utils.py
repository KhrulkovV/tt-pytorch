import numpy as np
import torch


def ind2sub(siz, idx):
    n = len(siz)
    b = len(idx)
    subs = []
    k = np.cumprod(siz[:-1])
    k = np.concatenate((np.ones(1), k))

    for i in range(n - 1, -1, -1):
        subs.append(torch.floor(idx.float() / k[i]))
        idx = torch.fmod(idx, k[i])

    return torch.stack(subs[::-1], dim=1)


def svd_fix(x):
    n = x.shape[0]
    m = x.shape[1]
    
    if n > m:
        u, s, v = torch.svd(x)
        
    else:
        u, s, v = torch.svd(x.t())
        v, u = u, v
    
    return u, s, v