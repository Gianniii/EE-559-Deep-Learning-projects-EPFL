# Authors: Gianni Lodetti, Luca Bracone, Omid Karimi

import torch
from models import *
from train import *
from torch import nn
from utils import generate_pair_sets
from torch import optim
import time


mini_batch_size = 100
nb_epochs = 25

def compute_nb_errors(model, input, target, mini_batch_size, with_auxiliary_loss):
    count = 0
    for b in range(0, train_input.size(0), mini_batch_size):
        if (with_auxiliary_loss == False) :
            output = model(input.narrow(0, b, mini_batch_size))
        else :
            _, _, output = model(input.narrow(0, b, mini_batch_size))

        _, predicted_classes = output.max(1)
        for k in range(mini_batch_size):
            if target[b + k] != predicted_classes[k]:
                count = count + 1
    return count

train_input, train_target, train_classes, test_input, \
test_target, test_classes = generate_pair_sets(1000)

#model = CNN()
#train_model(model, train_input, train_target, mini_batch_size)

model = SIAMESE_CNN_AUX()
start_time = time.time()
train_model_with_auxiliary_loss(model, train_input, train_target, train_classes, mini_batch_size)
print("training time: " + str(time.time() - start_time))
n = compute_nb_errors(model, test_input, test_target, mini_batch_size, True)
print(n)
