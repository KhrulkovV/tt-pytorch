import torch
import torch.nn as nn
import subprocess
import pandas as pd
import pickle


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    if len(preds.shape) == 1:
        rounded_preds = torch.round(torch.sigmoid(preds))
    else:
        rounded_preds = preds.argmax(1)
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum()/len(correct)
    return acc


def train(model, iterator, optimizer, criterion):

    epoch_loss = 0
    epoch_acc = 0
    total_len = 0

    model.train()
    
    if isinstance(criterion, nn.CrossEntropyLoss):
        dtype = torch.LongTensor
    elif isinstance(criterion, nn.BCEWithLogitsLoss):
        dtype = torch.FloatTensor

    for i, batch in enumerate(iterator):

        optimizer.zero_grad()
        device = batch.text.device
        labels = batch.label.type(dtype).to(device)
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, labels)
        acc = binary_accuracy(predictions, labels)
        loss.backward()
        optimizer.step()

        B = batch.label.shape[0]

        epoch_loss += B * loss.item()
        epoch_acc += B * acc.item()

        total_len += B


        if i > len(iterator):
            break

    return epoch_loss / total_len, epoch_acc / total_len


def evaluate(model, iterator, criterion):

    epoch_loss = 0
    epoch_acc = 0
    total_len = 0

    model.eval()
    
    if isinstance(criterion, nn.CrossEntropyLoss):
        dtype = torch.LongTensor
    elif isinstance(criterion, nn.BCEWithLogitsLoss):
        dtype = torch.FloatTensor

    with torch.no_grad():

        for i, batch in enumerate(iterator):
            
            device = batch.text.device
            labels = batch.label.type(dtype).to(device)
            predictions = model(batch.text).squeeze(1)

            loss = criterion(predictions, labels)

            acc = binary_accuracy(predictions, labels)
            B = batch.label.shape[0]

            epoch_loss += B * loss.item()
            epoch_acc += B * acc.item()
            total_len += B

            if i > len(iterator):
                break

    return epoch_loss / total_len, epoch_acc / total_len