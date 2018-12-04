from tttorch import TensorTrainBatch
import torch
import numpy as np


def gather_rows(mat, inds):
    """
    inds -- list of indices of shape batch_size x d
    d = len(mat.raw_shape[1])
    """
    cores = mat.tt_cores
    slices = []
    for k, core in enumerate(cores):
        i = inds[:, k]
        core = core.permute(1, 0, 2, 3)
        slices.append(torch.index_select(core, 0, i))

    return TensorTrainBatch(slices)
