import torch
import numpy as np
import torch.nn as nn
import t3nsor as t3


class TTEmbedding(nn.Module):
    def __init__(self, init=None, shape=None, tt_rank=2, stddev=1.0, batch_dim_last=True, permutation=None):
        super(TTEmbedding, self).__init__()

        if init is None:
            if shape is None:
                raise ValueError("if init is not provided, please specify shape")
        else:
            shape = init.raw_shape

        self.shape = shape

        if init is None:
            init = t3.random_matrix(shape, tt_rank=tt_rank, stddev=stddev)

        self.tt_matrix = init.to_parameter()
        self.parameters = self.tt_matrix.parameter
        self.batch_dim_last = batch_dim_last
        self.emb_shape = self.shape[0]
        self.permutation = permutation

    def forward(self, x):

        if self.batch_dim_last:
            x = x.permute(1, 0)

        batch_size = x.shape[0]        
        sent_size = x.shape[1]

        x = x.contiguous().view(-1)
        
        if self.permutation is not None:
            x = self.permutation[x]
        
        x_ind = t3.ind2sub(self.emb_shape, x).long()

        #x_ind = x_ind.flip(1)
        rows = t3.gather_rows(self.tt_matrix, x_ind).full()

        rows = rows.view(batch_size, sent_size, -1)
        if self.batch_dim_last:
            rows = rows.permute(1, 0, 2)

        return rows