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
        x = F.relu(self.bn1(self.fc1(x.view(-1, 392))))
        x = F.relu(self.bn2(self.fc2(x.view(-1, 100))))
        x = self.fc3(x.view(-1, 100))
        return x

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size = 5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3)
        self.fc1 = nn.Linear(256, 100)
        self.fc2 = nn.Linear(100, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size = 2, stride = 2))
        #print("a", x.size())
        x = self.bn1(x)
        #print("b", x.size())
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size = 2, stride = 1))
        #print("c", x.size())
        x = self.bn2(x)
        #print("d", x.size())
        x = F.relu(self.fc1(x.view(100, -1)))
        #print("d2", x.size())
        x = F.relu(self.fc2(x))
        #print("e", x.size())
        return x
