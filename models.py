import sys
import torch
import numpy as np
import torch.nn as nn
import t3nsor as t3


class LSTM_Classifier(nn.Module):
    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers,
        bidirectional,
        dropout
    ):
        super().__init__()
        self.rnn = nn.LSTM(
            embedding_dim, hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout)
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
            hidden = self.dropout(
                torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden)
        return self.fc(hidden.squeeze(0))

    