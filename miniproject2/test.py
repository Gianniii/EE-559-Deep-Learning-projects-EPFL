# Authors: Luca Bracone, Gianni Lodetti, Omid Karimi

import torch
import math
from modules import Linear, Sequential, ReLU, Tanh, MSELoss, Sigmoid
from optimizers import SGD

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

def train_model(model, train_input, train_target, mini_batch_size):
    criterion = MSELoss()
    optimizer = SGD(model.param())
    
    nb_epochs = 25
    for e in range(nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model.forward(train_input.narrow(0, b, mini_batch_size))
            #print("output" + str(output))
            loss = criterion.forward(output, train_target.narrow(0, b, mini_batch_size))
            model.zero_grad()
            
            loss_grad = criterion.backward()
            model.backward(loss_grad)
            
            optimizer.step()
            
def compute_nb_errors(model, input, target, mini_batch_size):
    error_count = 0
    for b in range(0, train_data.size(0), mini_batch_size):
        output = model.forward(input.narrow(0, b, mini_batch_size))
        #print(output)

        predicted_classes = torch.where(output < 0.5, 0, 1)
        #print(predicted_classes)
        for k in range(mini_batch_size):
            if target[b + k] != predicted_classes[k]:
                error_count += 1
    return error_count

#===============================================================================================

# TESTS
mini_batch_size = 100

model = Sequential([Linear(2, 25), Tanh(), Linear(25, 25), Tanh(), Linear(25, 25), Tanh(), Linear(25, 25), Tanh(), Linear(25, 1)])
train_model(model, train_data, train_target, mini_batch_size)
nb_errors = compute_nb_errors(model, test_data, test_target, mini_batch_size)
print("Number of errors: " + str(nb_errors))

