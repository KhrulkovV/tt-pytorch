import torch
import subprocess
import pandas as pd
import pickle


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum()/len(correct)
    return acc


def train(model, iterator, optimizer, criterion):

    epoch_loss = 0
    epoch_acc = 0

    model.train()


    for i, batch in enumerate(iterator):

        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

        if i > len(iterator):
            break


    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for i, batch in enumerate(iterator):


            predictions = model(batch.text).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

            if i > len(iterator):
                break

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate_embed(vocab, embeds):
    if len(vocab) < embeds.shape[0]:
        embeds = embeds[:len(vocab), :]
        
    d = dict(zip(list(vocab.keys()), list(embeds)))
    
    
    fin = '/workspace/tt-pytorch/tmp.pkl'
    fout = '/workspace/tt-pytorch/tmp.csv'
    
    
    with open(fin, 'wb') as f:
        pickle.dump(d, f)
        
    path = '/workspace/tt-pytorch/word-embeddings-benchmarks/scripts/evaluate_on_all.py'
    script_full = 'python {path} -f {fname} -o {outname}'.format(path=path, 
                                                                 fname=fin, 
                                                                 outname=fout)
    
    subprocess.run([script_full], shell=True)
    df = pd.read_csv(fout)

    return df



