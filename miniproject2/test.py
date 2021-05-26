# Authors: Luca Bracone, Gianni Lodetti, Omid Karimi

import torch
import math
import statistics
import time
from modules import Linear, Sequential, ReLU, Tanh, MSELoss, Sigmoid
from optimizers import SGD

torch.set_grad_enabled(False)

#=========================================================================================

# Data generation

# Randomly generate points and check wheter they are in a circle
def generate_data(n, center, radius):
    random_tensor = torch.empty((n, 2)).uniform_(0, 1)
    radius_sq = math.pow(radius, 2)
    
    temp_tensor = random_tensor.sub(center).pow(2).sum(1)
    target_tensor = torch.where(temp_tensor < radius_sq, 1, 0)
    
    return random_tensor, target_tensor

n = 1000
center = 0.5
radius = 1 / math.sqrt((2 * math.pi))

train_data, train_target = generate_data(n, center, radius)
test_data, test_target = generate_data(n, center, radius)

#==============================================================================================

# Training and error computation

def train_model(model, train_input, train_target, mini_batch_size, lossType = "MSE"):
    criterion = MSELoss()
    #if (lossType ==  "CrossEntropy"):
    #    criterion = CrossEntropyLoss()
    
    # keep track of loss during training
    log_losses = []

    optimizer = SGD(model.param())
    
    nb_epochs = 200
    for e in range(nb_epochs):
        mean_losses = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model.forward(train_input.narrow(0, b, mini_batch_size))
            
            loss = criterion.forward(output, train_target.narrow(0, b, mini_batch_size))
            mean_losses += loss.mean().item()
            
            model.zero_grad()
            
            loss_grad = criterion.backward()
            model.backward(loss_grad)
            
            optimizer.step()
            
        log_losses.append(mean_losses)
        
    return log_losses
            
def compute_nb_errors(model, input, target, mini_batch_size):
    error_count = 0
    # keep track of indices of wrong predictions for plot
    error_indices = []
    temp = input.clone()
    for b in range(0, train_data.size(0), mini_batch_size):
        output = model.forward(temp.narrow(0, b, mini_batch_size))

        predicted_classes = torch.where(output < 0.5, 0, 1)
        for k in range(mini_batch_size):
            if target[b + k] != predicted_classes[k]:
                error_indices.append((b+k))
                error_count += 1
    return error_count, error_indices

#===============================================================================================

# MAIN
mini_batch_size = 100
n = 1000
center = 0.5
radius = 1 / math.sqrt((2 * math.pi))

def run_model(model, nbr_runs):
    error_logs = []
    runtimes = []
    for i in range(nbr_runs):
        # Generate train and test data
        train_data, train_target = generate_data(n, center, radius)
        test_data, test_target = generate_data(n, center, radius)

        # Train model and compute train and test errors
        start_time = time.time()
        log_losses = train_model(model, train_data, train_target, mini_batch_size, "MSE")
        runtimes.append(time.time() - start_time)

        # Output statistics
        nb_errors, _ = compute_nb_errors(model, train_data, train_target, mini_batch_size)
        print("Training errors: " + str(nb_errors) + ", Training error rate: " + str((nb_errors * 100 / n)) + "%")
        nb_errors, error_indices = compute_nb_errors(model, test_data, test_target, mini_batch_size)
        print("Test errors: " + str(nb_errors) + ", Test error rate: " + str((nb_errors * 100 / n)) + "%")
        print("==================")
        error_logs.append(nb_errors)
    print(f"Mean of test errors over %d runs: %f" % (nbr_runs, statistics.mean(error_logs)))
    print(f"Standard deviation of test errors over %d runs: %f" % (nbr_runs, statistics.stdev(error_logs)))
    print(f"Average training time: %f sec" % (statistics.mean(runtimes)))
    print("==================\n")

# FINAL MODELS
    
print("================== MSE Loss ==================\n")
print("Model: 3 fully-connected layers with Tanh as activation function \n")
# Define 1st model
model = Sequential([Linear(2, 25), Tanh(), Linear(25, 25), Tanh(), Linear(25, 25), Tanh(), Linear(25, 25), Tanh(), Linear(25, 1), Sigmoid()])
run_model(model, 10)

print("Model: 3 fully-connected layers with ReLU as activation function \n")
# Define 2nd model
model = Sequential([Linear(2, 25, "He"), ReLU(), Linear(25, 25, "He"), ReLU(), Linear(25, 25, "He"), ReLU(), Linear(25, 25, "He"), ReLU(), Linear(25, 1), Sigmoid()])
run_model(model, 10)