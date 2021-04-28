from torch import nn
from torch.nn import functional as F

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(392, 400)
        self.fc2 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x.view(-1, 392)))
        x = F.relu(self.fc2(x.view(-1, 400)))
        x = self.fc3(x.view(-1, 20))
        return x
