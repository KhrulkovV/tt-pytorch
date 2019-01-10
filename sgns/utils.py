import re
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, DataLoader
from torch.utils.data.distributed import DistributedSampler


def create_vocabulary(sentences, r=200):
    prevocabulary = {}
    for sentence in sentences:
        for word in sentence:
            if word in prevocabulary:
                prevocabulary[word] += 1
            else:
                prevocabulary[word] = 1
    vocabulary = {}
    idx = 0
    for word in prevocabulary:
        if (prevocabulary[word] > r):
            vocabulary[word] = idx
            idx += 1
    return vocabulary


def create_triples_dataset(sentences, vocabulary, window_size=5):
    """Create training dataset consisting of triples (w, c, n_wc)"""
    dim = len(vocabulary)
    D = np.zeros((dim, dim))
    s = window_size//2
    for sentence in sentences:
        for i in range(len(sentence)):
            for j in range(max(0, i-s), min(i+s+1, len(sentence))):
                w, c = sentence[i], sentence[j]
                if (i != j) and (w in vocabulary) and (c in vocabulary):
                    c_idx = vocabulary[c]
                    w_idx = vocabulary[w]
                    D[c_idx][w_idx] += 1
    w, c = D.nonzero()
    data = {(w[i], c[i]): D[w[i], c[i]]for i in range(len(w))}
    unigram = np.power(D.sum(axis=0), 0.75)
    unigram = unigram / unigram.sum()
    return data, unigram


def load_corpus(data_dir="data", corpus="enwik8"):
    if os.path.isfile(data_dir+"/"+corpus+"_dataset.pkl"):
        with open(data_dir+"/"+corpus+"_vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        with open(data_dir+"/"+corpus+"_dataset.pkl", "rb") as f:
            dataset = pickle.load(f)
        unigram = np.load(data_dir+"/"+corpus+"_unigram.npz")["unigram"]
    else:
        file = open(data_dir+"/"+corpus+".txt", "r")
        docstr = "".join([line for line in file])
        sentences = re.split(r"[.!?]", docstr)
        sentences = [sentence.split()
                     for sentence in sentences if len(sentence) > 1]

        vocab = create_vocabulary(sentences, r=200)
        with open(data_dir+"/"+corpus+"_vocab.pkl", "wb") as f:
            pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)

        dataset, unigram = create_triples_dataset(sentences, vocab)
        with open(data_dir+"/"+corpus+"_dataset.pkl", "wb") as f:
            pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
        np.savez(data_dir+"/"+corpus+"_unigram.npz", unigram=unigram)

    return vocab, dataset, unigram


class CorpusDataset(Dataset):
    def __init__(
            self, dataset, vocab, unigram, neg_sampling_param=5):
        self.k = neg_sampling_param
        self.unigram = unigram
        self.uni_indices = np.arange(self.unigram.size)
        self.vocab_size = len(vocab)
        self.len = len(dataset)
        self.data = dataset
        self.w = [x[0] for x in self.data.keys()]
        self.c = [x[1] for x in self.data.keys()]
        self.update_negatives()

    def update_negatives(self):
        self.negatives = np.random.choice(
            self.uni_indices,
            size=self.k*self.len,
            p=self.unigram).reshape(self.len, self.k)

    def __getitem__(self, index):
        i, j = self.w[index], self.c[index]
        dct = {
            "word": i,
            "context": j,
            "count": self.data[(i, j)].astype(np.float32),
            "negatives": self.negatives[index]}
        return dct

    def __len__(self):
        return self.len


class CorpusSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.len = len(self.dataset)
        self.indices = np.arange(self.len)
        np.random.shuffle(self.indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return self.len
