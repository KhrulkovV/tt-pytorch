from t3nsor import TensorTrainBatch
from t3nsor import TensorTrain
import torch


def gather_rows(tt_mat, inds):
    """
    inds -- list of indices of shape batch_size x d
    d = len(tt_mat.raw_shape[1])
    """
    cores = tt_mat.tt_cores
    slices = []
    batch_size = int(inds[0].shape[0])


    ranks = [int(tt_core.shape[0]) for tt_core in tt_mat.tt_cores] + [1, ]


    for k, core in enumerate(cores):
        i = inds[k]
        #core = core.permute(1, 0, 2, 3).to(inds.device)

        cur_slice = torch.index_select(core, 1, i)

        if k == 0:
            res = cur_slice

        else:
            res = res.view(batch_size, -1, ranks[k])
            curr_core = cur_slice.view(ranks[k], batch_size, -1)
            res = torch.einsum('oqb,bow->oqw', (res, curr_core))


    return res





        #slices.append(torch.index_select(core, 1, i).permute(1, 0, 2, 3))



    return TensorTrainBatch(slices, convert_to_tensors=False)


def gather_rows_tt(tt_mat, inds):
    """
    inds -- list of indices of shape batch_size x d
    d = len(tt_mat.raw_shape[1])
    """
    cores = tt_mat.tt_cores
    slices = []
    batch_size = int(inds.shape[0])

    ranks = [int(cores.shape[0]) for cores in tt_mat.tt_cores] + [1, ]


    for k, core in enumerate(cores):
        i = inds[:, k]
        cur_slice = torch.index_select(core, 1, i)
        # r x B x M x r

        if k == 0:
            res = cur_slice.transpose(0, 1)
            # B x r x M x r

        else:
            res = res.contiguous().view(batch_size, -1, ranks[k])
            # B x rM x r
            curr_core = cur_slice.view(ranks[k], batch_size, -1)
            # r x B x Mr
            res = torch.einsum('oqb,bow->oqw', (res, curr_core))
            # B x rM x Mr

    res = torch.einsum('i...i->...', res.view(batch_size, ranks[0], res.shape[1] // ranks[0], -1, ranks[0]).transpose(0, 1))

    return res


def gather_rows_tr(tt_mat, inds):
    """
    inds -- list of indices of shape batch_size x d
    d = len(tt_mat.raw_shape[1])
    """
    cores = tt_mat.tr_cores
    slices = []
    batch_size = int(inds.shape[0])

    ranks = [int(cores.shape[0]) for cores in tt_mat.tr_cores] + [1, ]

    for k, core in enumerate(cores):
        i = inds[:, k]
        cur_slice = torch.index_select(core, 1, i)
        # r x B x M x r

        if k == 0:
            res = cur_slice.transpose(0, 1)
            # B x r x M x r

        else:
            res = res.contiguous().view(batch_size, -1, ranks[k])
            # B x rM x r
            curr_core = cur_slice.view(ranks[k], batch_size, -1)
            # r x B x Mr
            res = torch.einsum('oqb,bow->oqw', (res, curr_core))
            # B x rM x Mr

    res = torch.einsum('i...i->...', res.view(batch_size, ranks[0], res.shape[1] // ranks[0], -1, ranks[0]).transpose(0, 1))

    return res


def transpose(tt_matrix):
    cores = []
    for core in tt_matrix.tt_cores:
        cores.append(core.transpose(1, 2))
    return TensorTrain(cores)


def tt_dense_matmul(tt_matrix_a, matrix_b):
    """Multiplies a TT-matrix by a regular matrix, returns a regular matrix.
    Args:
    tt_matrix_a: `TensorTrain` object containing a TT-matrix of size M x N
    matrix_b: torch.Tensor of size N x P
    Returns
    torch.Tensor of size M x P
    """

    ndims = tt_matrix_a.ndims
    a_columns = tt_matrix_a.shape[1]
    b_rows = matrix_b.shape[0]
    if a_columns is not None and b_rows is not None:
        if a_columns != b_rows:
            raise ValueError('Arguments shapes should align got %d and %d instead.' %
                       (tt_matrix_a.shape, matrix_b.shape))

    a_shape = tt_matrix_a.shape
    a_raw_shape = tt_matrix_a.raw_shape
    b_shape = matrix_b.shape
    a_ranks = tt_matrix_a.ranks

    # If A is (i0, ..., id-1) x (j0, ..., jd-1) and B is (j0, ..., jd-1) x K,
    # data is (K, j0, ..., jd-2) x jd-1 x 1
    data = matrix_b.transpose(0, 1)
    data = data.view(-1, a_raw_shape[1][-1], 1)

    for core_idx in reversed(range(ndims)):
        curr_core = tt_matrix_a.tt_cores[core_idx]
        # On the k = core_idx iteration, after applying einsum the shape of data
        # becomes ik x (ik-1..., id-1, K, j0, ..., jk-1) x rank_k
        data = torch.einsum('aijb,rjb->ira', curr_core, data)
        if core_idx > 0:
          # After reshape the shape of data becomes
          # (ik, ..., id-1, K, j0, ..., jk-2) x jk-1 x rank_k
            new_data_shape = (-1, a_raw_shape[1][core_idx - 1], a_ranks[core_idx])
            data = data.contiguous().view(new_data_shape)

    # At the end the shape of the data is (i0, ..., id-1) x K
    return data.view(a_shape[0], b_shape[1])


def dense_tt_matmul(matrix_a, tt_matrix_b):
    ndims = tt_matrix_b.ndims
    a_columns = matrix_a.shape[1]
    b_rows = tt_matrix_b.shape[0]
    if a_columns is not None and b_rows is not None:
        if a_columns != b_rows:
            raise ValueError('Arguments shapes should align got %d and %d instead.' %
                             (matrix_a.shape, tt_matrix_b.shape))

    a_shape = matrix_a.shape
    b_shape = tt_matrix_b.shape
    b_raw_shape = tt_matrix_b.raw_shape
    data = matrix_a

    new_shape = [-1, ] + b_raw_shape[0] + [1, ]
    data = data.view(*new_shape)

    for core_idx in range(ndims):
        curr_core = tt_matrix_b.tt_cores[core_idx]
        data = torch.tensordot(data, curr_core, dims=[[1, -1], [1, 0]])

    return data.view(a_shape[0], b_shape[1])
