import torch
import numpy as np
import torch.nn as nn


class TensorTrain(object):
    def __init__(self, tt_cores, shape=None, tt_ranks=None, convert_to_tensors=True):
        #tt_cores = list(tt_cores)
        if convert_to_tensors:
            for i in range(len(tt_cores)):
                tt_cores[i] = torch.Tensor(tt_cores[i])

        self._tt_cores = tt_cores

        if len(self._tt_cores[0].shape) == 4:
            self._is_tt_matrix = True
        else:
            self._is_tt_matrix = False

        if self._is_tt_matrix:
            self._raw_shape = [[tt_core.shape[1] for tt_core in self._tt_cores],
                               [tt_core.shape[2] for tt_core in self._tt_cores]]
            self._shape = [int(np.prod(self._raw_shape[0])), int(np.prod(self._raw_shape[1]))]
            self._ndims = len(self._raw_shape[0])

        else:
            self._raw_shape = [tt_core.shape[1] for tt_core in self._tt_cores]
            self._shape = [tt_core.shape[1] for tt_core in self._tt_cores]
            self._ndims = len(self._raw_shape)

        self._ranks = [tt_core.shape[0] for tt_core in self._tt_cores] + [1, ]
        self._is_parameter = False
        self._parameter = None
        self._dof = np.sum([np.prod(list(tt_core.shape)) for tt_core in self._tt_cores])
        self._total = np.prod(self._shape)
        

    @property
    def tt_cores(self):
        """A list of TT-cores.
        Returns:
          A list of 3d or 4d tensors of shape
        """
        return self._tt_cores

    @property
    def raw_shape(self):
        return self._raw_shape

    @property
    def is_tt_matrix(self):
        return self._is_tt_matrix

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
        

    def to(self, device):
        new_cores = []
        for core in self.tt_cores:
            new_cores.append(core.to(device))
        return TensorTrain(new_cores, convert_to_tensors=False)

    def detach(self):
        new_cores = []
        for core in self.tt_cores:
            new_cores.append(core.detach())
        return TensorTrain(new_cores, convert_to_tensors=False)

    def requires_grad_(self, requires_grad=True):
        new_cores = []
        for core in self.tt_cores:
            new_cores.append(core.requires_grad_(requires_grad))
        return TensorTrain(new_cores, convert_to_tensors=False)

    def to_parameter(self):
        new_cores = []
        for core in self.tt_cores:
            core = nn.Parameter(core)
            core.is_tt = True
            new_cores.append(core)

        tt_p = TensorTrain(new_cores, convert_to_tensors=False)
        tt_p._parameter = nn.ParameterList(tt_p.tt_cores)        
        tt_p._is_parameter = True
        return tt_p

    def full(self):
        num_dims = self.ndims
        ranks = self.ranks
        shape = self.shape
        raw_shape = self.raw_shape
        res = self.tt_cores[0]

        for i in range(1, num_dims):
            res = res.view(-1, ranks[i])
            curr_core = self.tt_cores[i].view(ranks[i], -1)
            res = torch.matmul(res, curr_core)

        if self.is_tt_matrix:
            intermediate_shape = []
            for i in range(num_dims):
                intermediate_shape.append(raw_shape[0][i])
                intermediate_shape.append(raw_shape[1][i])

            res = res.view(*intermediate_shape)
            transpose = []
            for i in range(0, 2 * num_dims, 2):
                transpose.append(i)
            for i in range(1, 2 * num_dims, 2):
                transpose.append(i)
            res = res.permute(*transpose)
        
        if self.is_tt_matrix:
            res = res.contiguous().view(*shape)
        else:
            res = res.view(*shape)
        return res

    def __str__(self):
        """A string describing the TensorTrain object, its TT-rank, and shape."""
        shape = self.shape
        tt_ranks = self.ranks
        device = self.tt_cores[0].device
        compression_rate = self.total / self.dof
        if self.is_tt_matrix:
            raw_shape = self.raw_shape
            return "A TT-Matrix of size %d x %d, underlying tensor" \
                   "shape: %s x %s, TT-ranks: %s " \
                   "\n on device '%s' with compression rate %.2f" % (shape[0], shape[1],
                                           raw_shape[0], raw_shape[1],
                                           tt_ranks, device, compression_rate)
        else:
            return "A Tensor Train of shape %s, TT-ranks: %s" \
                   "\n on device '%s' with compression rate %.2f" % (shape, tt_ranks, device, compression_rate)


class TensorTrainBatch():
    def __init__(self, tt_cores, shape=None, tt_ranks=None, convert_to_tensors=True):
        #tt_cores = list(tt_cores)
        if convert_to_tensors:
            for i in range(len(tt_cores)):
                tt_cores[i] = torch.Tensor(tt_cores[i])

        self._tt_cores = tt_cores

        self._batch_size = self._tt_cores[0].shape[0]

        if len(self._tt_cores[0].shape) == 5:
            self._is_tt_matrix = True
        else:
            self._is_tt_matrix = False

        if self._is_tt_matrix:
            self._raw_shape = [[tt_core.shape[2] for tt_core in self._tt_cores],
                               [tt_core.shape[3] for tt_core in self._tt_cores]]
            self._shape = [self._batch_size, int(
                np.prod(self._raw_shape[0])), int(np.prod(self._raw_shape[1]))]
            self._ndims = len(self._raw_shape[0])

        else:
            self._raw_shape = [tt_core.shape[2] for tt_core in self._tt_cores]
            self._shape = [self._batch_size, ] + [tt_core.shape[2] for tt_core in self._tt_cores]
            self._ndims = len(self._raw_shape)

        self._ranks = [tt_core.shape[1] for tt_core in self._tt_cores] + [1, ]

    @property
    def tt_cores(self):
        """A list of TT-cores.
        Returns:
          A list of 4d or 5d tensors.
        """
        return self._tt_cores

    @property
    def raw_shape(self):
        return self._raw_shape

    @property
    def is_tt_matrix(self):
        return self._is_tt_matrix

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
    def batch_size(self):
        return self._batch_size

    def to(self, device):
        new_cores = []
        for core in self.tt_cores:
            new_cores.append(core.to(device))
        return TensorTrainBatch(new_cores, convert_to_tensors=False)

    def detach(self):
        new_cores = []
        for core in self.tt_cores:
            new_cores.append(core.detach())
        return TensorTrainBatch(new_cores, convert_to_tensors=False)

    def requires_grad_(self, requires_grad=True):
        new_cores = []
        for core in self.tt_cores:
            new_cores.append(core.requires_grad_(requires_grad))
        return TensorTrainBatch(new_cores, convert_to_tensors=False)

    def full(self):
        num_dims = self.ndims
        ranks = self.ranks
        shape = self.shape
        raw_shape = self.raw_shape
        res = self.tt_cores[0]
        batch_size = self.batch_size

        for i in range(1, num_dims):
            res = res.view(batch_size, -1, ranks[i])
            curr_core = self.tt_cores[i].view(batch_size, ranks[i], -1)
            res = torch.einsum('oqb,obw->oqw', (res, curr_core))

        if self.is_tt_matrix:
            intermediate_shape = [batch_size]
            for i in range(num_dims):
                intermediate_shape.append(raw_shape[0][i])
                intermediate_shape.append(raw_shape[1][i])
            res = res.view(*intermediate_shape)
            transpose = [0]
            for i in range(0, 2 * num_dims, 2):
                transpose.append(i + 1)
            for i in range(1, 2 * num_dims, 2):
                transpose.append(i + 1)
            res = res.permute(transpose)
            
        if self.is_tt_matrix:           
            res = res.contiguous().view(*shape)
        else:
            res = res.view(*shape)
        return res

    def __str__(self):
        """A string describing the TensorTrainBatch, its TT-rank and shape."""
        shape = self.shape
        tt_ranks = self.ranks
        batch_size_str = str(self.batch_size)
        device = self.tt_cores[0].device

        if self.is_tt_matrix:
            raw_shape = self.raw_shape
            type_str = 'TT-matrices'

            return "A %s element batch of %s of size %d x %d, underlying tensor " \
                   "shape: %s x %s, TT-ranks: %s" \
                   "on device '%s' " % (batch_size_str, type_str,
                                        shape[1], shape[2],
                                        raw_shape[0], raw_shape[1],
                                        tt_ranks, device)
        else:
            type_str = 'Tensor Trains'
            return "A %s element batch of %s of shape %s, TT-ranks: %s \n on device '%s'" % \
                   (batch_size_str, type_str, shape[1:], tt_ranks, device)
