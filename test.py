# Authors: Gianni Lodetti, Luca Bracone, Omid Karimi

import torch
from models import *
from train import *
from utils import generate_pair_sets
from torch import nn
from torch import optim
import statistics
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

"""train_input, train_target, train_classes, test_input, \
test_target, test_classes = generate_pair_sets(1000)"""

# Example output for best model

runtimes = []
error_logs = []
nbr_runs = 10
print("=========== Siamese convolutional neural network ===========\n")
for i in range(nbr_runs):
    aux_loss = True
    model = SIAMESE_CNN_AUX(act = "leaky")
    # Generate new data at each iteration
    train_input, train_target, train_classes, test_input, \
    test_target, test_classes = generate_pair_sets(1000)
    
    # Train model
    start_time = time.time()
    train_model(model, train_input, train_target, train_classes, mini_batch_size, aux_loss, opt = "Adam")
    runtime = time.time() - start_time
    runtimes.append(runtime)
    print("Iteration " + str(i + 1) + ": Training time: " + str(runtime))
    
    # Evaluate model
    nbr_errors = compute_nb_errors(model, test_input, test_target, mini_batch_size, aux_loss)
    error_logs.append(nbr_errors)
    print("Test errors: " + str(nbr_errors) + ", Test error rate: " + str((nbr_errors * 100 / 1000)) + "%")
    
print("\n==================") 
print(f"Mean of test errors over %d runs: %f" % (nbr_runs, statistics.mean(error_logs)))
print(f"Standard deviation of test errors over %d runs: %f" % (nbr_runs, statistics.stdev(error_logs)))
print(f"Average training time: %f sec" % (statistics.mean(runtimes)))
print("==================\n") 
