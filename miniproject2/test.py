# Authors: Luca Bracone, Gianni Lodetti, Omid Karimi

import torch
import math
from modules import *
from optimizers import *

torch.set_grad_enabled(False)

#=========================================================================================

# Data generation

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

mini_batch_size = 100
nb_epochs = 25

def train_model(model, train_input, train_target, mini_batch_size):
    criterion = modules.MSELoss()
    #optimizer = optim.SGD(model.parameters(), lr = 1e-1)
    for e in range(nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))

            model.zero_grad()
            loss.backward()
            optimizer.step()

def compute_nb_errors(model, input, target, mini_batch_size, with_auxiliary_loss):
    error_count = 0
    for b in range(0,train_input.size(0), mini_batch_size):
        if with_auxiliary_loss:
            _, _, output = model(input.narrow(0, b, mini_batch_size))
        else:
            output = model(input.narrow(0, b, mini_batch_size))

        _, predicted_classes = output.max(1)
        for k in range(mini_batch_size):
            if target[b + k] != predicted_classes[k]:
                error_count += 1
    return error_count

#===============================================================================================

# YEP
