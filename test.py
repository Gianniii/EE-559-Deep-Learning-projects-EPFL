import torch
from torch import optim


mini_batch_size = 100
nb_epochs = 25

def compute_nb_errors(model, input, target, mini_batch_size):
    count = 0
    for b in range(0,train_input.size(0), mini_batch_size):
        output = model(input.narrow(0, b, mini_batch_size))
        _, predicted_classes = output.max(1)
        for k in range(mini_batch_size):
            if target[b + k] != predicted_classes[k]:
                count = count + 1
    return count

def train_model(model, train_input, train_target, mini_batch_size):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = 1e-1)
    for e in range(nb_epochs):
        # We do this with mini-batches
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))

            model.zero_grad()
            loss.backward()
            optimizer.step()



train_input, train_target, train_classes, test_input, \
test_target, test_classes = generate_pair_sets(1000)

model = NeuralNet()
train_model(model, train_input, train_target, mini_batch_size)
n = compute_nb_errors(model, test_input, test_target, mini_batch_size )
print(n)
