import numpy as np
import torch

from t3nsor.tensor_ring import TensorRing
# from t3nsor.tensor_ring import TensorRingBatch


def _validate_input_parameters_tr(is_tensor, shape, **params):
    """Internal function for validating input parameters
    Args:
      is_tensor: bool, determines whether we attempt to construct a TT-tensor or
        a TT-matrix (needed for the correct shape checks).
      shape: array, the desired shape of the generated TT object
      params: optional, possible values:
        batch_size: int, for constructing batches
        tr_rank: array or int, desired TT-ranks
    """

    if is_tensor:
        if len(shape.shape) != 1:
            raise ValueError('shape should be 1d array, got %a' % shape)
        if np.any(shape < 1):
            raise ValueError('all elements in `shape` should be positive, got %a' %
                             shape)
        if not all(isinstance(sh, np.integer) for sh in shape):
            raise ValueError('all elements in `shape` should be integers, got %a' %
                             shape)
    else:
        if len(shape.shape) != 2:
            raise ValueError('shape should be 2d array, got %a' % shape)
        if shape[0].size != shape[1].size:
            raise ValueError('shape[0] should have the same length as shape[1], but'
                             'got %d and %d' % (shape[0].size, shape[1].size))
        if np.any(shape.flatten() < 1):
            raise ValueError('all elements in `shape` should be positive, got %a' %
                             shape)
        if not all(isinstance(sh, np.integer) for sh in shape.flatten()):
            raise ValueError('all elements in `shape` should be integers, got %a' %
                             shape)

    if 'batch_size' in params:
        batch_size = params['batch_size']
        if not isinstance(batch_size, (int, np.integer)):
            raise ValueError('`batch_size` should be integer, got %f' % batch_size)
        if batch_size < 1:
            raise ValueError('Batch size should be positive, got %d' % batch_size)
    if 'tr_rank' in params:
        tr_rank = params['tr_rank']
        if tr_rank.size == 1:
            if not isinstance(tr_rank[()], np.integer):
                raise ValueError('`tr_rank` should be integer, got %f' % tr_rank[()])
        if tr_rank.size > 1:
            if not all(isinstance(tt_r, np.integer) for tt_r in tr_rank):
                raise ValueError('all elements in `tr_rank` should be integers, got'
                                 ' %a' % tr_rank)
        if np.any(tr_rank < 1):
            raise ValueError('`tr_rank` should be positive, got %a' % tr_rank)

        if is_tensor:
            if tr_rank.size != 1 and tr_rank.size != (shape.size + 1):
                raise ValueError('`tr_rank` array has inappropriate size, expected'
                                 '1 or %d, got %d' % (shape.size + 1, tr_rank.size))
        else:
            if tr_rank.size != 1 and tr_rank.size != (shape[0].size + 1):
                raise ValueError('`tr_rank` array has inappropriate size, expected'
                                 '1 or %d, got %d' % (shape[0].size + 1, tr_rank.size))


def tensor_ones_tr(shape, dtype=torch.float32):
    """Generate TR-tensor of the given shape with all entries equal to 1.
    Args:
      shape: array representing the shape of the future tensor
      dtype: [torch.float32] dtype of the resulting tensor.
      name: string, name of the Op.
    Returns:
      TensorRing object containing a TT-tensor
    """

    shape = np.array(shape)
    _validate_input_parameters_tr(is_tensor=True, shape=shape)
    num_dims = shape.size
    tr_cores = num_dims * [None]
    for i in range(num_dims):
        curr_core_shape = (1, shape[i], 1)
        tr_cores[i] = torch.ones(curr_core_shape, dtype=dtype)

    return TensorRing(tr_cores)


def tensor_zeros_tr(shape, dtype=torch.float32):
    """Generate TT-tensor of the given shape with all entries equal to 0.
    Args:
      shape: array representing the shape of the future tensor
      dtype: [torch.float32] dtype of the resulting tensor.
      name: string, name of the Op.
    Returns:
      TensorRing object containing a TT-tensor
    """

    shape = np.array(shape)
    _validate_input_parameters_tr(is_tensor=True, shape=shape)
    num_dims = shape.size

    tr_cores = num_dims * [None]

    for i in range(num_dims):
        curr_core_shape = (1, shape[i], 1)
        tr_cores[i] = torch.zeros(curr_core_shape, dtype=dtype)

    return TensorRing(tr_cores)


def matrix_zeros_tr(shape, rank=2, dtype=torch.float32):
    """Generate TR-matrix of the given shape with all entries equal to 0.
    Args:
      shape: array representing the shape of the future tensor
      dtype: [torch.float32] dtype of the resulting tensor.
      name: string, name of the Op.
    Returns:
      TensorRing object containing a TT-matrix
    """

    shape = np.array(shape)
    _validate_input_parameters_tr(is_tensor=False, shape=shape)
    num_dims = shape[0].size

    tr_cores = num_dims * [None]

    curr_core_shape = (1, shape[0][0], shape[1][0], rank)
    tr_cores[0] = torch.zeros(curr_core_shape, dtype=dtype)
    
    for i in range(1, num_dims - 1):
        curr_core_shape = (rank, shape[0][i], shape[1][i], rank)
        tr_cores[i] = torch.zeros(curr_core_shape, dtype=dtype)
        
    curr_core_shape = (rank, shape[0][num_dims - 1], shape[1][num_dims - 1], 1)
    tr_cores[num_dims - 1] = torch.zeros(curr_core_shape, dtype=dtype)    

    return TensorRing(tr_cores)


def eye_tr(shape, dtype=torch.float32):
    """Creates an identity TR-matrix.
    Args:
      shape: array which defines the shape of the matrix row and column
        indices.
      dtype: [torch.float32] dtype of the resulting matrix.
      name: string, name of the Op.
    Returns:
      TensorRing containing an identity TT-matrix of size
      np.prod(shape) x np.prod(shape)
    """
    shape = np.array(shape)
    # In this special case shape is in the same format as in the TR-tensor case
    _validate_input_parameters_tr(is_tensor=True, shape=shape)

    num_dims = shape.size
    tr_cores = num_dims * [None]
    for i in range(num_dims):
        curr_core_shape = (1, int(shape[i]), int(shape[i]), 1)
        tr_cores[i] = torch.eye_tr(shape[i], dtype=dtype).view(*curr_core_shape)

    return TensorRing(tr_cores)


def matrix_with_random_cores_tr(shape, tr_rank=2, mean=0., stddev=1.,
                             dtype=torch.float32):
    """Generate a TT-matrix of given shape with N(mean, stddev^2) cores.
    Args:
      shape: 2d array, shape[0] is the shape of the matrix row-index,
        shape[1] is the shape of the column index.
        shape[0] and shape[1] should have the same number of elements (d)
        Also supports omitting one of the dimensions for vectors, e.g.
          matrix_with_random_cores_tr([[2, 2, 2], None])
        and
          matrix_with_random_cores_tr([None, [2, 2, 2]])
        will create an 8-element column and row vectors correspondingly.
      tr_rank: a number or a (d+1)-element array with ranks.
      mean: a number, the mean of the normal distribution used for
        initializing TT-cores.
      stddev: a number, the standard deviation of the normal distribution used
        for initializing TT-cores.
      dtype: [tf.float32] dtype of the resulting matrix.
      name: string, name of the Op.
    Returns:
      TensorRing containing a TT-matrix of size
        np.prod(shape[0]) x np.prod(shape[1])
    """
    # TODO: good distribution to init training.
    # In case the shape is immutable.
    shape = list(shape)
    # In case shape represents a vector, e.g. [None, [2, 2, 2]]
    if shape[0] is None:
        shape[0] = np.ones(len(shape[1]), dtype=int)
    # In case shape represents a vector, e.g. [[2, 2, 2], None]
    if shape[1] is None:
        shape[1] = np.ones(len(shape[0]), dtype=int)
    shape = np.array(shape)
    tr_rank = np.array(tr_rank)
    _validate_input_parameters_tr(is_tensor=False, shape=shape, tr_rank=tr_rank)

    num_dims = shape[0].size
    if tr_rank.size == 1:
        tr_rank = tr_rank * np.ones(num_dims - 1)
        tr_rank = np.concatenate([[1], tr_rank, [1]])

    tr_rank = tr_rank.astype(int)
    tr_cores = [None] * num_dims

    for i in range(num_dims):
        curr_core_shape = (tr_rank[i], shape[0][i], shape[1][i],
                           tr_rank[i + 1])
        tr_cores[i] = torch.randn(curr_core_shape, dtype=dtype) * stddev + mean

    return TensorRing(tr_cores)


def random_matrix_tr(shape, tr_rank=2, mean=0., stddev=1.,
                  dtype=torch.float32):
    """Generate a random TT-matrix of the given shape with given mean and stddev.
    Entries of the generated matrix (in the full format) will be iid and satisfy
    E[x_{i1i2..id}] = mean, Var[x_{i1i2..id}] = stddev^2, but the distribution is
    in fact not Gaussian.
    In the current implementation only mean 0 is supported. To get
    a random_matrix_tr with specified mean but tr_rank greater by 1 you can call
    x = ttt.random_matrix_tr(shape, tr_rank, stddev=stddev)
    x = mean * t3f.ones_like(x) + x
    Args:
      shape: 2d array, shape[0] is the shape of the matrix row-index,
        shape[1] is the shape of the column index.
        shape[0] and shape[1] should have the same number of elements (d)
        Also supports omitting one of the dimensions for vectors, e.g.
          random_matrix_tr([[2, 2, 2], None])
        and
          random_matrix_tr([None, [2, 2, 2]])
        will create an 8-element column and row vectors correspondingly.
      tr_rank: a number or a (d+1)-element array with ranks.
      mean: a number, the desired mean for the distribution of entries.
      stddev: a number, the desired standard deviation for the distribution of
        entries.
      dtype: [tf.float32] dtype of the resulting matrix.
      name: string, name of the Op.
    Returns:
      TensorRing containing a TT-matrix of size
        np.prod(shape[0]) x np.prod(shape[1])
    """
    # TODO: good distribution to init training.
    # In case the shape is immutable.
    shape = list(shape)
    # In case shape represents a vector, e.g. [None, [2, 2, 2]]
    if shape[0] is None:
        shape[0] = np.ones(len(shape[1]), dtype=int)
    # In case shape represents a vector, e.g. [[2, 2, 2], None]
    if shape[1] is None:
        shape[1] = np.ones(len(shape[0]), dtype=int)
    shape = np.array(shape)
    tr_rank = np.array(tr_rank)

    _validate_input_parameters_tr(is_tensor=False, shape=shape, tr_rank=tr_rank)

    num_dims = shape[0].size
    if tr_rank.size == 1:
        tr_rank = tr_rank * np.ones(num_dims + 1)
        #tr_rank = np.concatenate([[1], tr_rank, [1]])

    tr_rank = tr_rank.astype(int)
    var = np.prod(tr_rank)

    # Empirically entries of a TT tensor with cores initialized from N(0, 1)
    # will have variances np.prod(tr_rank) and mean 0.
    # We scale each TT-core to obtain the desired stddev

    cr_exponent = -1.0 / (2 * num_dims)
    var = np.prod(tr_rank ** cr_exponent)
    core_stddev = stddev ** (1.0 / num_dims) * var

    tt = matrix_with_random_cores_tr(shape, tr_rank=tr_rank, stddev=core_stddev, dtype=dtype)

    if np.abs(mean) < 1e-8:
        return tt
    else:
        raise NotImplementedError('non-zero mean is not supported yet')

        
        
def glorot_initializer_tr(shape, tr_rank=2, dtype=torch.float32):
    shape = list(shape)
    # In case shape represents a vector, e.g. [None, [2, 2, 2]]
    if shape[0] is None:
        shape[0] = np.ones(len(shape[1]), dtype=int)
      # In case shape represents a vector, e.g. [[2, 2, 2], None]
    if shape[1] is None:
        shape[1] = np.ones(len(shape[0]), dtype=int)
    shape = np.array(shape)
    tr_rank = np.array(tr_rank)
    _validate_input_parameters_tr(is_tensor=False, shape=shape, tr_rank=tr_rank)
    n_in = np.prod(shape[0])
    n_out = np.prod(shape[1])
    lamb = 2.0 / (n_in + n_out)
    return random_matrix_tr(shape, tr_rank=tr_rank, stddev=np.sqrt(lamb),dtype=dtype)



def matrix_batch_with_random_cores_tr(shape, batch_size=1, tr_rank=2, mean=0., stddev=1.,
                             dtype=torch.float32):
    """Generate a TT-matrix of given shape with N(mean, stddev^2) cores.
    Args:
      shape: 2d array, shape[0] is the shape of the matrix row-index,
        shape[1] is the shape of the column index.
        shape[0] and shape[1] should have the same number of elements (d)
        Also supports omitting one of the dimensions for vectors, e.g.
          matrix_with_random_cores_tr([[2, 2, 2], None])
        and
          matrix_with_random_cores_tr([None, [2, 2, 2]])
        will create an 8-element column and row vectors correspondingly.
      tr_rank: a number or a (d+1)-element array with ranks.
      mean: a number, the mean of the normal distribution used for
        initializing TT-cores.
      stddev: a number, the standard deviation of the normal distribution used
        for initializing TT-cores.
      dtype: [tf.float32] dtype of the resulting matrix.
      name: string, name of the Op.
    Returns:
      TensorRing containing a TT-matrix of size
        np.prod(shape[0]) x np.prod(shape[1])
    """
    # TODO: good distribution to init training.
    # In case the shape is immutable.
    shape = list(shape)
    # In case shape represents a vector, e.g. [None, [2, 2, 2]]
    if shape[0] is None:
        shape[0] = np.ones(len(shape[1]), dtype=int)
    # In case shape represents a vector, e.g. [[2, 2, 2], None]
    if shape[1] is None:
        shape[1] = np.ones(len(shape[0]), dtype=int)
    shape = np.array(shape)
    tr_rank = np.array(tr_rank)
    _validate_input_parameters_tr(is_tensor=False, shape=shape, tr_rank=tr_rank, batch_size=batch_size)

    num_dims = shape[0].size
    if tr_rank.size == 1:
        tr_rank = tr_rank * np.ones(num_dims - 1)
        tr_rank = np.concatenate([[1], tr_rank, [1]])

    tr_rank = tr_rank.astype(int)
    tr_cores = [None] * num_dims

    for i in range(num_dims):
        curr_core_shape = (batch_size, tr_rank[i], shape[0][i], shape[1][i],
                           tr_rank[i + 1])
        tr_cores[i] = torch.randn(curr_core_shape, dtype=dtype) * stddev + mean

    return TensorRingBatch(tr_cores)


def random_matrix_tr_batch_tr(shape, batch_size=1, tr_rank=2, mean=0., stddev=1.,
                  dtype=torch.float32):
    """Generate a random TT-matrix of the given shape with given mean and stddev.
    Entries of the generated matrix (in the full format) will be iid and satisfy
    E[x_{i1i2..id}] = mean, Var[x_{i1i2..id}] = stddev^2, but the distribution is
    in fact not Gaussian.
    In the current implementation only mean 0 is supported. To get
    a random_matrix_tr with specified mean but tr_rank greater by 1 you can call
    x = ttt.random_matrix_tr(shape, tr_rank, stddev=stddev)
    x = mean * t3f.ones_like(x) + x
    Args:
      shape: 2d array, shape[0] is the shape of the matrix row-index,
        shape[1] is the shape of the column index.
        shape[0] and shape[1] should have the same number of elements (d)
        Also supports omitting one of the dimensions for vectors, e.g.
          random_matrix_tr([[2, 2, 2], None])
        and
          random_matrix_tr([None, [2, 2, 2]])
        will create an 8-element column and row vectors correspondingly.
      tr_rank: a number or a (d+1)-element array with ranks.
      mean: a number, the desired mean for the distribution of entries.
      stddev: a number, the desired standard deviation for the distribution of
        entries.
      dtype: [tf.float32] dtype of the resulting matrix.
      name: string, name of the Op.
    Returns:
      TensorRing containing a TT-matrix of size
        np.prod(shape[0]) x np.prod(shape[1])
    """
    # TODO: good distribution to init training.
    # In case the shape is immutable.
    shape = list(shape)
    # In case shape represents a vector, e.g. [None, [2, 2, 2]]
    if shape[0] is None:
        shape[0] = np.ones(len(shape[1]), dtype=int)
    # In case shape represents a vector, e.g. [[2, 2, 2], None]
    if shape[1] is None:
        shape[1] = np.ones(len(shape[0]), dtype=int)
    shape = np.array(shape)
    tr_rank = np.array(tr_rank)

    _validate_input_parameters_tr(is_tensor=False, shape=shape, tr_rank=tr_rank, batch_size=batch_size)

    num_dims = shape[0].size
    if tr_rank.size == 1:
        tr_rank = tr_rank * np.ones(num_dims - 1)
        tr_rank = np.concatenate([[1], tr_rank, [1]])

    tr_rank = tr_rank.astype(int)
    var = np.prod(tr_rank)

    # Empirically entries of a TT tensor with cores initialized from N(0, 1)
    # will have variances np.prod(tr_rank) and mean 0.
    # We scale each TT-core to obtain the desired stddev

    cr_exponent = -1.0 / (2 * num_dims)
    var = np.prod(tr_rank ** cr_exponent)
    core_stddev = stddev ** (1.0 / num_dims) * var

    tt = matrix_batch_with_random_cores_tr(shape, batch_size=batch_size, tr_rank=tr_rank, stddev=core_stddev, dtype=dtype)

    if np.abs(mean) < 1e-8:
        return tt
    else:
        raise NotImplementedError('non-zero mean is not supported yet')