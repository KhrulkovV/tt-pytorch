import numpy as np
import torch

from t3nsor.tensor_train import TensorTrain
from t3nsor.utils import svd_fix


def to_tt_tensor(tens, max_tt_rank=10, epsilon=None):

    shape = tens.shape
    d = len(shape)
    max_tt_rank = np.array(max_tt_rank).astype(np.int32)
    if max_tt_rank.size == 1:
        max_tt_rank = [int(max_tt_rank), ] * (d+1)

    ranks = [1] * (d + 1)
    tt_cores = []

    for core_idx in range(d - 1):
        curr_mode = shape[core_idx]
        rows = ranks[core_idx] * curr_mode
        tens = tens.view(rows, -1)
        columns = tens.shape[1]
        u, s, v = svd_fix(tens)
        if max_tt_rank[core_idx + 1] == 1:
            ranks[core_idx + 1] = 1
        else:
            ranks[core_idx + 1] = min(max_tt_rank[core_idx + 1], rows, columns)

        u = u[:, 0:ranks[core_idx + 1]]
        s = s[0:ranks[core_idx + 1]]
        v = v[:, 0:ranks[core_idx + 1]]
        core_shape = (ranks[core_idx], curr_mode, ranks[core_idx + 1])
        tt_cores.append(u.view(*core_shape))
        tens = torch.matmul(torch.diag(s), v.permute(1, 0))

    last_mode = shape[-1]

    core_shape = (ranks[d - 1], last_mode, ranks[d])
    tt_cores.append(tens.view(core_shape))

    return TensorTrain(tt_cores, convert_to_tensors=False)


def to_tt_matrix(mat, shape, max_tt_rank=10, epsilon=None):

    shape = list(shape)
    # In case shape represents a vector, e.g. [None, [2, 2, 2]]
    if shape[0] is None:
        shape[0] = np.ones(len(shape[1])).astype(int)
    # In case shape represents a vector, e.g. [[2, 2, 2], None]
    if shape[1] is None:
        shape[1] = np.ones(len(shape[0])).astype(int)

    shape = np.array(shape)

    def np2int(x):
        return list(map(int, x))

    tens = mat.view(*np2int(shape.flatten()))
    d = len(shape[0])
    # transpose_idx = 0, d, 1, d+1 ...
    transpose_idx = np.arange(2 * d).reshape(2, d).T.flatten()
    transpose_idx = np2int(transpose_idx)
    tens = tens.permute(*transpose_idx)
    new_shape = np2int(np.prod(shape, axis=0))
    tens = tens.contiguous().view(*new_shape)
    tt_tens = to_tt_tensor(tens, max_tt_rank, epsilon)
    tt_cores = []
    tt_ranks = tt_tens.ranks
    for core_idx in range(d):
        curr_core = tt_tens.tt_cores[core_idx]
        curr_rank = tt_ranks[core_idx]
        next_rank = tt_ranks[core_idx + 1]
        curr_core_new_shape = (curr_rank, shape[0, core_idx], shape[1, core_idx], next_rank)
        curr_core = curr_core.view(*curr_core_new_shape)
        tt_cores.append(curr_core)
    return TensorTrain(tt_cores, convert_to_tensors=False)
