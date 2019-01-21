import argparse
import sys
sys.path.insert(0, '..')


parser = argparse.ArgumentParser()
parser.add_argument('--use_tt', type=bool, default=False)
parser.add_argument('--ranks', type=int, default=8)
parser.add_argument('--d', type=int, default=3)
parser.add_argument('--embed_dim', type=int)
parser.add_argument('--voc_dim', default=25000, type=int)
parser.add_argument('--lr', default=5e-4)
parser.add_argument('--gpu', default='', type=str)
parser.add_argument('--hidden_dim', default=128, type=int)
parser.add_argument('--n_epochs',  default=10, type=int)
parser.add_argument('--fout',  default=None, type=str)
parser.add_argument('--dropout', default=0.5, type=float)



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
import random


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


TEXT = data.Field(tokenize='spacy', fix_length=1000)

LABEL = data.LabelField(dtype=torch.float)

print('Building dataset...')
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
print('Done')

test_data_list = list(test_data)

random.shuffle(test_data_list)

test_data_ = test_data_list[:12500]
val_data_ = test_data_list[12500:]


def sort_key(ex):
    return len(ex.text)

val_dataset = data.dataset.Dataset(val_data_, fields=[('text', TEXT), ('label', LABEL)])
test_dataset = data.dataset.Dataset(test_data_, fields=[('text', TEXT), ('label', LABEL)])




TEXT.build_vocab(train_data, max_size=args.voc_dim - 2)
LABEL.build_vocab(train_data)

BATCH_SIZE = 256

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, val_dataset, test_dataset),
    batch_size=BATCH_SIZE,
    device=device)


valid_iterator.sort_key = sort_key
test_iterator.sort_key = sort_key


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = args.embed_dim
HIDDEN_DIM = args.hidden_dim
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = args.dropout

actual_vocab_size = len(TEXT.vocab.stoi)


lstm_model = LSTM_Classifier(embedding_dim=EMBEDDING_DIM,
                             hidden_dim=HIDDEN_DIM,
                             output_dim=OUTPUT_DIM,
                             n_layers=N_LAYERS,
                             bidirectional=BIDIRECTIONAL,
                             dropout=DROPOUT)

if args.use_tt:
        embed_model = t3.TTEmbedding(voc_size=INPUT_DIM, emb_size=EMBEDDING_DIM, auto_shapes=True, d=args.d, 
                                     tt_rank=args.ranks, padding_idx=1)

        compression_rate = INPUT_DIM * EMBEDDING_DIM / embed_model.tt_matrix.dof
else:
    embed_model = nn.Embedding(num_embeddings=INPUT_DIM, embedding_dim=EMBEDDING_DIM)

    compression_rate = 1.0
    
    
    
model = nn.Sequential(embed_model, lstm_model)


criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
print(model)
N_EPOCHS = args.n_epochs

log = {'compression_rate':compression_rate, 'train_loss':[], 'test_loss':[], 'train_acc':[], 'test_acc':[], 'valid_acc':[], 'valid_loss':[]}

for epoch in range(N_EPOCHS):

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    log['train_loss'].append(train_loss)
    log['test_loss'].append(test_loss)
    log['train_acc'].append(train_acc)
    log['test_acc'].append(test_acc)
    log['valid_acc'].append(valid_acc)
    log['valid_loss'].append(valid_loss)

    if args.fout is not None:
        with open(args.fout, 'wb') as f:
            pickle.dump(log, f)
    print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {test_loss:.3f} | Val. Acc: {test_acc*100:.2f}% |')
