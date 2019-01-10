import argparse
import os

import time
import torch 
import numpy as np
import torch.nn as nn
import t3nsor as t3
import torch.optim as optim

from apex.parallel import DistributedDataParallel as DDP
from apex.fp16_utils import to_python_float

import torch.utils.data
import torch.utils.data.distributed
import torch.distributed as dist

from sgns.utils import *
from sgns.models import *

def log1p_exp(input_tensor):
    """ Numerically stable function for computing log(1+exp(x)).
    """
    x = input_tensor * input_tensor.ge(0).to(torch.float32)
    res = x + torch.log1p(
        torch.exp(-torch.abs(input_tensor)))
    return res

# Training settings
parser = argparse.ArgumentParser(description='PyTorch word2vec Example')
parser.add_argument(
    '--batch-size', type=int, default=512, metavar='N',
    help='input batch size for training (default: 512)')
parser.add_argument(
    '--embedding-dim', type=int, default=256, metavar='N',
    help='embedding dimensionality (default: 256)')
parser.add_argument(
    '--epochs', type=int, default=50, metavar='N',
    help='number of epochs to train (default: 50)')
parser.add_argument(
    '--lr', type=float, default=0.001, metavar='LR',
    help='learning rate (default: 0.001)')
parser.add_argument(
    '--no-cuda', action='store_true', default=False,
    help='disables CUDA training')
parser.add_argument(
    '--log-interval', type=int, default=200, metavar='N',
    help='how many batches to wait before logging training status')
parser.add_argument(
    '--save-interval', type=int, default=10, metavar='N',
    help='how many epochs to wait between checkpoints')
parser.add_argument("--logdir", type=str, default=None)
parser.add_argument("--local_rank", default=0, type=int)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
os.makedirs(args.logdir, exist_ok=True)


'''Add a convenience flag to see if we are running distributed'''
args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1

if args.distributed:
    '''Check that we are running with cuda, as distributed is only supported for cuda.'''
    assert args.cuda, "Distributed mode requires running with CUDA."

    '''
    Set cuda device so everything is done on the right GPU.
    THIS MUST BE DONE AS SOON AS POSSIBLE.
    '''
    torch.cuda.set_device(args.local_rank)

    '''Initialize distributed communication'''
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

print ("Loading dataset...")
vocab, triples, unigram = load_corpus(corpus="enwik9")
dataset = CorpusDataset(triples, vocab, unigram)
print ("Dataset loaded.")
vocab_size = len(vocab)
embedding_dim = args.embedding_dim
batch_size = args.batch_size

emb_model = VanillaEmbeddings(vocab_size, embedding_dim)
if args.cuda:
    emb_model.cuda()
if args.distributed:
    emb_model = DDP(emb_model)
    
optimizer = optim.Adam(emb_model.parameters(), lr=args.lr)
sgns_param = 5
epoch_losses = []


def train(epoch, loader):
    losses = []
    emb_model.train()
    for batch_idx, batch in enumerate(loader):
        words, contexts, counts, negatives = \
        batch["word"], batch["context"], \
        batch["count"], batch["negatives"]
        words = torch.LongTensor(words)[:, None]
        contexts = torch.LongTensor(contexts)[:, None]
        counts = torch.FloatTensor(counts)[:, None]
        negatives = torch.LongTensor(negatives)
        if args.cuda:
            words = words.cuda()
            contexts = contexts.cuda()
            counts = counts.cuda()
            negatives = negatives.cuda()
        
        optimizer.zero_grad()
        all_contexts = torch.cat((contexts, negatives), dim=1)
        w_emb, c_emb = emb_model(words, all_contexts)
        wc = torch.einsum("bi,bji->bj", (w_emb, c_emb))
        pos_wc, neg_wc = wc[:, :1], wc[:, 1:]
        pos_term = log1p_exp(-pos_wc)
        neg_term = sgns_param * log1p_exp(neg_wc).mean(dim=1, keepdim=True)
        loss = torch.mean(counts * (pos_term + neg_term))
        loss.backward()
        optimizer.step()
    
        loss_ = to_python_float(loss.data)
        losses.append(loss_)
        if batch_idx % args.log_interval == 0 and args.local_rank == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch, batch_idx * len(words), len(loader.dataset),
                100. * batch_idx / len(loader), loss_))
    return np.mean(losses)


for epoch in range(1, args.epochs+1):
    start_time = time.time()

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        train_sampler.set_epoch(epoch)
    #else:
    #    sampler = CorpusSampler(dataset)

        train_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=args.cuda,
        sampler=train_sampler)
    
    avg_loss = train(epoch, train_loader)
    epoch_losses.append(avg_loss)

    if epoch % 1 == 0:
        print (f"Epoch {epoch}:")
        print (f"Loss: {avg_loss}")
        time_elapsed = (time.time() - start_time) / 60
        print (f"Time elapsed: {time_elapsed} minutes")
        print ("---------------------")
        np.savez(f"{args.logdir}/loss_{args.local_rank}.npz", loss=epoch_losses)

    if epoch % args.save_interval == 0:
        torch.save(
            emb_model.state_dict(),
            f"{args.logdir}/model_{epoch // args.save_interval}.pth.tar")

    dataset.update_negatives()
