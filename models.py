from torch import nn
from torch.nn import functional as F

class BasicNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(392, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 2)
        self.bn1 = nn.BatchNorm1d(100)
        self.bn2 = nn.BatchNorm1d(100)

    def forward(self, x):
        x = F.relu(self.fc1(x.view(-1, 392)))
        x = self.bn1(x)
        x = F.relu(self.fc2(x.view(-1, 100)))
        x = self.bn2(x)
        x = self.fc3(x.view(-1, 100))
        return x

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(2, 32, kernel_size = 5)
        self.c2 = nn.Conv2d(32, 64, kernel_size = 5)
        self.fc3 = nn.Linear(100, 2)
        self.bn1 = nn.BatchNorm1d(100)
        self.bn2 = nn.BatchNorm1d(100)

    def forward(self, x):
        x = F.relu(self.fc1(x.view(-1, 392)))
        x = self.bn1(x)
        x = F.relu(self.fc2(x.view(-1, 100)))
        x = self.bn2(x)
        x = self.fc3(x.view(-1, 100))
        return x
