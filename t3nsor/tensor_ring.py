import torch
import numpy as np
import torch.nn as nn
import t3nsor as t3

class TensorRing(object):
    def __init__(self, tr_cores, shape=None, tr_ranks=None, convert_to_tensors=True):
        #tr_cores = list(tr_cores)
        if convert_to_tensors:
            for i in range(len(tr_cores)):
                tr_cores[i] = torch.Tensor(tr_cores[i])

        self._tr_cores = tr_cores

        if len(self._tr_cores[0].shape) == 4:
            self._is_tr_matrix = True
        else:
            self._is_tr_matrix = False

        if self._is_tr_matrix:
            self._raw_shape = [[tr_core.shape[1] for tr_core in self._tr_cores],
                               [tr_core.shape[2] for tr_core in self._tr_cores]]
            self._shape = [int(np.prod(self._raw_shape[0])), int(np.prod(self._raw_shape[1]))]
            self._ndims = len(self._raw_shape[0])

        else:
            self._raw_shape = [tr_core.shape[1] for tr_core in self._tr_cores]
            self._shape = [tr_core.shape[1] for tr_core in self._tr_cores]
            self._ndims = len(self._raw_shape)

        self._ranks = [tr_core.shape[0] for tr_core in self._tr_cores] + [1, ]
        self._is_parameter = False
        self._parameter = None
        self._dof = np.sum([np.prod(list(tr_core.shape)) for tr_core in self._tr_cores])
        self._total = np.prod(self._shape)

    @property
    def tr_cores(self):
        """A list of TR-cores.
        Returns:
          A list of 3d or 4d tensors of shape
        """
        return self._tr_cores

    @property
    def cores(self):
        """A list of TR-cores.
        Returns:
          A list of 3d or 4d tensors of shape
        """
        return self._tr_cores

    @property
    def raw_shape(self):
        return self._raw_shape

    @property
    def is_tr_matrix(self):
        return self._is_tr_matrix

    @property
    def shape(self):
        return self._shape

    @property
    def ranks(self):
        return self._ranks

    @property
    def ndims(self):
        return self._ndims

    @property
    def is_parameter(self):
        return self._is_parameter

    @property
    def parameter(self):
        if self.is_parameter:
            return self._parameter
        else:
            raise ValueError('Not a parameter, run .to_parameter() first')

    @property
    def dof(self):
        return self._dof

    @property
    def total(self):
        return self._total


    def full(self):
        num_dims = self._ndims
        ranks = self._ranks
        shape = self._shape
        raw_shape = self._raw_shape
        res = self.tr_cores[0]

        for core_idx in range(1, num_dims):
            curr_core = self.tr_cores[core_idx]
#             print('loop', core_idx, curr_core.shape)
            res = torch.tensordot(res, curr_core, dims=[[-1], [0]])

        res = torch.einsum('i...i->...', res) # trace
#         print(res.shape)

        if self.is_tr_matrix:
            transpose = []
            for i in range(0, 2 * num_dims, 2):
                transpose.append(i)
            for i in range(1, 2 * num_dims, 2):
                transpose.append(i)
            res = res.permute(*transpose)
#         print(transpose)
        if self.is_tr_matrix:
            res = res.contiguous().view(*shape)
        else:
            res = res.view(*shape)
        return res

    def to_parameter(self):
        new_cores = []
        for core in self.tr_cores:
            core = nn.Parameter(core)
            core.is_tr = True
            new_cores.append(core)

        tr_p = t3.TensorRing(new_cores, convert_to_tensors=False)
        tr_p._parameter = nn.ParameterList(tr_p.tr_cores)
        tr_p._is_parameter = True
        return tr_p
