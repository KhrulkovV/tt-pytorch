import torch
import numpy as np
import torch.nn as nn
import t3nsor as t3


class TTEmbedding(nn.Module):
    def __init__(self, 
                 init=None, 
                 shape=None, 
                 voc_size=None,
                 emb_size=None,
                 auto_shapes=None,
                 d=3,
                 tt_rank=8, 
                 stddev=None,
                 batch_dim_last=None, 
                 padding_idx=None):
        
        super(TTEmbedding, self).__init__()

        if auto_shapes:
            voc_quantization =  t3.utils.suggest_tt(voc_size, d=d)
            emb_quantization = t3.utils.get_tt_shape(emb_size, d=d)
            
            shape = [voc_quantization, emb_quantization]
            
        
        if init is None:
            if shape is None:
                raise ValueError("if init is not provided, please specify shape")
        else:
            shape = init.raw_shape

        self.shape = shape

        if init is None:
            init = t3.glorot_initializer(shape, tt_rank=tt_rank)

        self.tt_matrix = init.to_parameter()
        self.parameters = self.tt_matrix.parameter
        
        #for p in self.parameters():
        #    p.name = 'tt_core'
        
        self.batch_dim_last = batch_dim_last
        self.voc_size = int(np.prod(self.shape[0]))
        self.emb_size = int(np.prod(self.shape[1]))
        
        self.voc_quant = self.shape[0]
        self.emb_quant = self.shape[1]
        
        self.padding_idx = padding_idx

    def forward(self, x):
        
        xshape = list(x.shape)
        xshape_new = xshape + [self.emb_size,]
        x = x.contiguous().view(-1)
          
        x_ind = t3.ind2sub(self.voc_quant, x).long()
        rows = t3.gather_rows(self.tt_matrix, x_ind).full()
                 
        rows = rows.view(x.shape[0], -1)
        
        if self.padding_idx is not None:
            rows = torch.where(x.view(-1, 1) != self.padding_idx, rows, torch.zeros_like(rows))
                 
        
        rows = rows.view(*xshape_new)
        
        return rows