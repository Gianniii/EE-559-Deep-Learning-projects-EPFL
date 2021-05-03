import torch
from models import *
from torch import nn
from torch import optim

mini_batch_size = 100
nb_epochs = 25

def train_model_with_auxiliary_loss(model, train_input, train_target, train_classes, mini_batch_size):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = 1e-1)
    for e in range(nb_epochs):
        # We do this with mini-batches
        for b in range(0, train_input.size(0), mini_batch_size):
            img1, img2, biggest = model(train_input.narrow(0, b, mini_batch_size))
            
            loss1 = criterion(img1, train_classes.narrow(0, b, mini_batch_size).narrow(1,0,1).view(-1))
            loss2 = criterion(img2, train_classes.narrow(0, b, mini_batch_size).narrow(1,1,1).view(-1))
            loss3 = criterion(biggest, train_target.narrow(0, b, mini_batch_size))
            
            #add loss1 and loss2 as auxiliary losses
            loss = loss1 + loss2 + loss3

            model.zero_grad()
            loss.backward()
            optimizer.step()
        print(e, loss.item())