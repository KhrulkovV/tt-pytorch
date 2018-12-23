import sys
sys.path.append("../")
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import t3nsor as t3


def log1p_exp(input_tensor):
    """ Computationally stable function for computing log(1+exp(x)).
    """
    x = input_tensor * input_tensor.ge(0).to(torch.float32)
    res = x + torch.log1p(
        torch.exp(-torch.abs(input_tensor)))
    return res


class VanillaEmbeddings(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        """Class for vanilla word2vec embedding model."""
        super().__init__()
        self.embedding_dim = embedding_dim
        self.w_emb = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim)
        self.c_emb = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim)
        self.init_embeddings()

    def init_embeddings(self):
        """Initialize embedding weight like word2vec.
        The w_emb is a uniform distribution in
        [-0.5/em_size, 0.5/emb_size], and the elements
        of c_embedding are zeroes.
        """
        initrange = 0.5 / self.embedding_dim
        self.w_emb.weight.data.uniform_(-initrange, initrange)
        self.c_emb.weight.data.uniform_(-0, 0)

    def forward(self, word_indices, context_indices):
        w = self.w_emb.forward(word_indices)
        c = self.c_emb.forward(context_indices)
        return w, c


class TTEmbeddings(nn.Module):
    def __init__(
            self,
            vocab_size,
            embedding_dim,
            tt_shape,
            tt_rank=32,
            stddev=None):
        """Class for TT word2vec embedding model."""
        super().__init__()
        self.embedding_dim = embedding_dim
        assert vocab_size <= np.prod(tt_shape[0])
        stddev = stddev or 0.5 / self.embedding_dim
        self.w_emb = t3.TTEmbedding(
            shape=tt_shape,
            stddev=stddev,
            tt_rank=tt_rank,
            batch_dim_last=False)
        self.c_emb = t3.TTEmbedding(
            shape=tt_shape,
            stddev=1e-6,
            tt_rank=tt_rank,
            batch_dim_last=False)

    def init_embeddings(self, tt_matrices):
        """Initialize embedding with pretrained tt-matrices.
        tt_matrices = [w_tt_matrix, c_tt_matrix].
        """
        w, c = tt_matrices
        self.w_emb = t3.TTEmbedding(init=w, batch_dim_last=False)
        self.c_emb = t3.TTEmbedding(init=c, batch_dim_last=False)

    def forward(self, word_indices, context_indices):
        w = self.w_emb.forward(word_indices[:, None])[:, 0, :]
        c = self.c_emb.forward(context_indices[:, None])[:, 0, :]
        return w, c


class Word2VecSGNS:
    def __init__(
            self,
            embeddings_model,
            neg_sampling_param=5,
            learning_rate=1e-3):
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.emb_model = embeddings_model.to(self._device)
        self.optimizer = optim.Adam(
            self.emb_model.parameters(), lr=learning_rate)
        self.k = neg_sampling_param

    def train(self, batch):
        words, contexts, counts, negatives = \
            batch["word"], batch["context"], \
            batch["count"], batch["negative"]
        words = torch.LongTensor(words).to(self._device)
        contexts = torch.LongTensor(contexts).to(self._device)
        counts = torch.FloatTensor(counts).to(self._device)
        negatives = torch.FloatTensor(negatives).to(self._device)

        w_emb, c_emb = self.emb_model(words, contexts)
        wc = torch.einsum("bi,bi->b", (w_emb, c_emb))
        loss = (log1p_exp(-wc) * counts + \
            self.k * negatives * log1p_exp(wc)).mean()
        
        # update embeddings parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
