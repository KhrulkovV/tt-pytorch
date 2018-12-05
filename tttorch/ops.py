from tttorch import TensorTrainBatch
import torch


def gather_rows(tt_mat, inds):
    """
    inds -- list of indices of shape batch_size x d
    d = len(tt_mat.raw_shape[1])
    """
    cores = tt_mat.tt_cores
    slices = []
    for k, core in enumerate(cores):
        i = inds[:, k]
        core = core.permute(1, 0, 2, 3)
        slices.append(torch.index_select(core, 0, i))

    return TensorTrainBatch(slices, convert_to_tensors=False)
