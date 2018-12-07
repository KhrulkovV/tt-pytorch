import argparse
import sys
sys.path.insert(0, '..')


parser = argparse.ArgumentParser()
parser.add_argument('--use_tt', type=bool, default=True)
parser.add_argument('--ranks', type=int, default=8)
parser.add_argument('--voc_shape', default=[], type=int, nargs='+')
parser.add_argument('--embed_shape', default=[], type=int, nargs='+')
parser.add_argument('--lr', default=1e-3)
parser.add_argument('--gpu', default='', type=str)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--max_len', default=None, type=int)
parser.add_argument('--bidirectional', default=True, type=bool)
parser.add_argument('--dropout', default=0.5, type=float)
parser.add_argument('--n_layers', default=2, type=int)
parser.add_argument('--hidden_dim', default=256, type=int)
parser.add_argument('--n_epochs',  default=10, type=int)
parser.add_argument('--fout',  default=None, type=str)



args = parser.parse_args()


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu


import torch
import numpy as np
import torch.nn as nn
import t3nsor as t3
from torchtext import data
from torchtext import datasets
import torch.optim as optim
from models import LSTM_Classifier
from utils import binary_accuracy, train, evaluate
import pickle


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.max_len is None:
    TEXT = data.Field(tokenize='spacy')
else:
    TEXT = data.Field(tokenize='spacy', fix_length=args.max_len)

LABEL = data.LabelField(dtype=torch.float)

print('Building dataset...')
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
print('Done')

raw_voc_size = int(np.prod(args.voc_shape))
raw_embed_shape = int(np.prod(args.embed_shape))

TEXT.build_vocab(train_data, max_size=raw_voc_size - 2)
LABEL.build_vocab(train_data)

BATCH_SIZE = args.batch_size

train_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, test_data),
    batch_size=BATCH_SIZE,
    device=device)

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = raw_embed_shape
HIDDEN_DIM = args.hidden_dim
OUTPUT_DIM = 1
N_LAYERS = args.n_layers
BIDIRECTIONAL = args.bidirectional
DROPOUT = args.dropout


lstm_model = LSTM_Classifier(embedding_dim=EMBEDDING_DIM,
                             hidden_dim=HIDDEN_DIM,
                             output_dim=OUTPUT_DIM,
                             n_layers=N_LAYERS,
                             bidirectional=BIDIRECTIONAL,
                             dropout=DROPOUT)

if args.use_tt:
    embed_model = t3.TTEmbedding(shape=[args.voc_shape, args.embed_shape],
                                 tt_rank=args.ranks,
                                 batch_dim_last=True)

else:
    embed_model = nn.Embedding(num_embeddings=INPUT_DIM, embedding_dim=EMBEDDING_DIM)

model = nn.Sequential(embed_model, lstm_model)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)


N_EPOCHS = args.n_epochs

log = {'train_loss':[], 'test_loss':[], 'train_acc':[], 'test_acc':[]}

for epoch in range(N_EPOCHS):

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    test_loss, test_acc = evaluate(model, test_iterator, criterion)

    log['train_loss'].append(train_loss)
    log['test_loss'].append(test_loss)
    log['train_acc'].append(train_acc)
    log['test_acc'].append(test_acc)

    if args.fout is not None:
        with open(args.fout, 'wb') as f:
            pickle.dump(log, f)
    print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {test_loss:.3f} | Val. Acc: {test_acc*100:.2f}% |')
