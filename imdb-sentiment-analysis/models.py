import sys
sys.path.insert(0, '..')
import torch
import numpy as np
import torch.nn as nn
import t3nsor as t3


class LSTM_Classifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
        if bidirectional:
            self.fc = nn.Linear(hidden_dim * 2, output_dim)
        else:
            self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional

    def forward(self, x):
        embedded = self.dropout(x)
        output, (hidden, cell) = self.rnn(embedded)
        if self.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden)
        return self.fc(hidden.squeeze(0))

    
class SkipgramNegSamplingTT(nn.Module):
    
    def __init__(self, vocab_size, projection_dim):
        super(SkipgramNegSamplingTT, self).__init__()
        
        self.embedding_v = t3.TTEmbedding(init=t3.glorot_initializer(shape=[vocab_size, projection_dim], tt_rank=16),
                                          batch_dim_last=False)

        self.embedding_u = t3.TTEmbedding(shape=[vocab_size, projection_dim], tt_rank=16, stddev=0.0001, batch_dim_last=False)
        
        self.logsigmoid = nn.LogSigmoid()
                        
    def forward(self, center_words, target_words, negative_words):
        center_embeds = self.embedding_v(center_words) # B x 1 x D
        target_embeds = self.embedding_u(target_words) # B x 1 x D
        
        neg_embeds = -self.embedding_u(negative_words) # B x K x D
        
        positive_score = target_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2) # Bx1
        negative_score = torch.sum(neg_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2), 1).view(negs.size(0), -1) # BxK -> Bx1
        
        loss = self.logsigmoid(positive_score) + self.logsigmoid(negative_score)
        
        return -torch.mean(loss)
    
    def prediction(self, inputs):
        embeds = self.embedding_v(inputs)
        
        return embeds    