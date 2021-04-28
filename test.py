import torch
from torch import nn
from torch.nn import functional as F

def generate_pair_sets(nb):
    if args.data_dir is not None:
        data_dir = args.data_dir
    else:
        data_dir = os.environ.get('PYTORCH_DATA_DIR')
        if data_dir is None:
            data_dir = './data'

    train_set = datasets.MNIST(data_dir + '/mnist/', train = True, download = True)
    train_input = train_set.data.view(-1, 1, 28, 28).float()
    train_target = train_set.targets

    test_set = datasets.MNIST(data_dir + '/mnist/', train = False, download = True)
    test_input = test_set.data.view(-1, 1, 28, 28).float()
    test_target = test_set.targets

    return mnist_to_pairs(nb, train_input, train_target) + \
           mnist_to_pairs(nb, test_input, test_target)

class NeuralNet(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(392, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x.view(-1, 392)))
        x = F.relu(self.fc2(x.view(-1, 200)))
        x = self.fc3(x.view(-1, 100))
        return x

criterion = nn.MSELoss()
eta, mini_batch_size = 1e-1, 100
nb_epochs = 25

def compute_nb_errors(model, input, target, mini_batch_size):
    count = 0
    output = model(input)
    for b in range(output.size(0)):
        _, j = output[b,:].max(0)
        if target[b,j] < 0.5:
            count += 1
    return count

def train_model(model, train_input, train_target, mini_batch_size):
    for e in range(nb_epochs):
        acc_loss = 0
        # We do this with mini-batches
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            acc_loss = acc_loss + loss.item()

            model.zero_grad()
            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= eta * p.grad

        if e % 20 == 0:
            print(e, acc_loss)
