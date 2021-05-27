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

import copy
import statistics

# Runs given model nbr_runs time and outputs statistics
def run_model(model, aux_loss, opt, nbr_runs):
    error_logs = []
    runtimes = []
    for i in range(nbr_runs):
        # Generate data at eah iteration
        train_input, train_target, train_classes, test_input, \
        test_target, test_classes = generate_pair_sets(1000)
        
        temp_model = copy.deepcopy(model)
        start_time = time.time()
        train_model(temp_model, train_input, train_target, train_classes, mini_batch_size, aux_loss, opt=opt)
        runtime = start_time - time.time()
        runtimes.append(runtime)
        print("Training time: " + str(runtime))
        
        n = compute_nb_errors(temp_model, test_input, test_target, mini_batch_size, aux_loss)
        error_logs.append(n)
        print("Iteration: "+ str(i + 1) + ", Test errors: " + str(n) + "Test error rate: " + str((n * 100 / n)) + "%")
        
    print(f"Mean of test errors over %d runs: %f" % (nbr_runs, statistics.mean(error_logs)))
    print(f"Standard deviation of test errors over %d runs: %f" % (nbr_runs, statistics.stdev(error_logs)))
    print(f"Average training time: %f sec" % (statistics.mean(runtimes)))

# DEFINE MODELS TO TEST

model = SIAMESE_CNN_AUX(act = "leaky")
run_model(model, aux_loss=True, opt="SGD", nbr_runs=10)


for i in range(10):
    aux_loss = True
    model = SIAMESE_CNN_AUX(act = "leaky")
    start_time = time.time()
    train_model(model, train_input, train_target, train_classes, mini_batch_size, aux_loss, opt = "Adam")
    print("training time: " + str(time.time() - start_time))
    n = compute_nb_errors(model, test_input, test_target, mini_batch_size, aux_loss)
    print("interation: "+ str(i) + " nb errors: " + str(n)) 

