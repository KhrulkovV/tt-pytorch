from t3nsor import TensorTrainBatch
from t3nsor import TensorTrain
from torch.autograd import Function
import torch

def gather_rows(tt_mat, inds):
    """
    inds -- list of indices of shape batch_size x d
    d = len(tt_mat.raw_shape[1])
    """
    cores = tt_mat.cores
    slices = []
    batch_size = int(inds.shape[0])

    ranks = [int(cores.shape[0]) for cores in tt_mat.cores] + [1, ]

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
    res = torch.einsum('i...i->...', res.view(batch_size, ranks[0], res.shape[1] // ranks[0], -1, ranks[0]).transpose(0, 1))

    return res

def transpose(tt_matrix):
    cores = []
    for core in tt_matrix.tt_cores:
        cores.append(core.transpose(1, 2))
    return TensorTrain(cores)

def transpose_cores(tt_cores):
    cores = []
    for core in tt_cores:
        cores.append(core.transpose(1, 2))
    return cores


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

def naive_dense_tt_matmul(matrix_a, tt_matrix_b):
    ndims = tt_matrix_b.ndims
    a_columns = matrix_a.shape[1]
    b_rows = tt_matrix_b.shape[0]
    if a_columns is not None and b_rows is not None:
        if a_columns != b_rows:
            raise ValueError('Arguments shapes should align got %d and %d instead.' %
                             (matrix_a.shape, tt_matrix_b.shape))

    assert ndims == 3

    core0 = tt_matrix_b.tt_cores[0]  # 1 x n x m x r
    core1 = tt_matrix_b.tt_cores[1]  # r x n x m x r
    core2 = tt_matrix_b.tt_cores[2]  # r x n x m x 1

    input = matrix_a.view(-1, core0.shape[1], core1.shape[1], core2.shape[1])
    B = input.shape[0]

    full = torch.einsum('abcd,defg,ghij->bcefhi', core0, core1, core2)
    res = torch.einsum('abcd,bqcsdx->aqsx', input, full)
    return res.contiguous().view(B, -1)


def _naive_dense_tt_cores(matrix_a, tt_cores):


    core0 = tt_cores[0]  # 1 x n x m x r
    core1 = tt_cores[1]  # r x n x m x r
    core2 = tt_cores[2]  # r x n x m x 1

    input = matrix_a.view(-1, core0.shape[1], core1.shape[1], core2.shape[1])
    B = input.shape[0]

    full = torch.einsum('abcd,defg,ghij->bcefhi', core0, core1, core2)
    res = torch.einsum('abcd,bqcsdx->aqsx', input, full)
    return res.contiguous().view(B, -1)




def naive_full(tt_a):
    ndims = tt_a.ndims
    assert ndims == 3
    try:
        # TT-Embedding
        core0, core1, core2 = tt_a.tt_cores
    except:
        # TR-Embedding
        core0, core1, core2 = tt_a.tr_cores

    full = torch.einsum('abcd,defg,ghia->bcefhi', core0, core1, core2)
    full = full.reshape(tt_a.shape[0], tt_a.shape[1])
    return full

def naive_dense_tr_matmul(matrix_a, tr_matrix_b):
    ndims = tr_matrix_b.ndims
    a_columns = matrix_a.shape[1]
    b_rows = tr_matrix_b.shape[0]
    if a_columns is not None and b_rows is not None:
        if a_columns != b_rows:
            raise ValueError('Arguments shapes should align got %d and %d instead.' %
                             (matrix_a.shape, tr_matrix_b.shape))

    assert ndims == 3

    core0 = tr_matrix_b.tr_cores[0]  # 1 x n x m x r
    core1 = tr_matrix_b.tr_cores[1]  # r x n x m x r
    core2 = tr_matrix_b.tr_cores[2]  # r x n x m x 1

    input = matrix_a.view(-1, core0.shape[1], core1.shape[1], core2.shape[1])
    B = input.shape[0]

    full = torch.einsum('abcd,defg,ghia->bcefhi', core0, core1, core2)
    res = torch.einsum('abcd,bqcsdx->aqsx', input, full)
    return res.contiguous().view(B, -1)


class TTLinearFunction(Function):
    """Multiplies a TT-matrix (weights) by a regular matrix (input), returns a regular matrix.
    Args:
    weight: `TensorTrain` object containing a TT-matrix of size M x N
    input: torch.Tensor of size N x P
    Returns
    torch.Tensor of size M x P
    """
    @staticmethod
    def _dtdc0(dydt_reshaped, cores):
        return torch.einsum('ijkabc,xjbt,tkcs->siax', dydt_reshaped, cores[1], cores[2])

    @staticmethod
    def _dtdc1(dydt_reshaped, cores):
        # dydt_reshape (m1 x m2 x m3) x (n1 x n2 x n3)
        # output: r1 x m2 x n2 x r2
        # core0: 1 x m1 x n1 x r1
        # core2: r2 x m3 x n3 x 1

        return torch.einsum('ijkabc,xiat,pkcx->tjbp', dydt_reshaped, cores[0], cores[2])
    @staticmethod
    def _dtdc2(dydt_reshaped, cores):

        # dydt_reshape (m1 x m2 x m3) x (n1 x n2 x n3)
        # output: r2 x m3 x n3 x 1
        # core0: 1 x m1 x n1 x r1
        # core1: r1 x m2 x n2 x r2

        return torch.einsum('ijkabc,xiat,tjbs->skcx', dydt_reshaped, cores[0], cores[1])


    @staticmethod
    def forward(ctx, input, *tt_cores):
        ctx.save_for_backward(input, *tt_cores)
        data = _naive_dense_tt_cores(input, tt_cores)
        return data

    @staticmethod
    def backward(ctx, grad_output):
        inp, *tt_cores = ctx.saved_tensors
        grad_input = grad_c0 = grad_c1 = grad_c2 = None
        grad_input = _naive_dense_tt_cores(grad_output, transpose_cores(tt_cores))
        dydt = inp.t() @ grad_output
        shape = [core.shape[1] for core in tt_cores] + [core.shape[2] for core in tt_cores]
        dydt_reshaped = dydt.view(*shape)
        grad_c0 = TTLinearFunction._dtdc0(dydt_reshaped, tt_cores)
        grad_c1 = TTLinearFunction._dtdc1(dydt_reshaped, tt_cores)
        grad_c2 = TTLinearFunction._dtdc2(dydt_reshaped, tt_cores)

        return grad_input, grad_c0, grad_c1, grad_c2
