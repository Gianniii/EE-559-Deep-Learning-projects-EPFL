# Authors: Gianni Lodetti, Luca Bracone, Omid Karimi
# Definition of neural networks

import torch
from torch import nn
from torch.nn import functional as F

# Basic Neural Network, 3 fully-connected layers with batch normalization
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


# Convolutional Neural Network, 2 convolutional layers and 2 fully-connected layers
class CNN(nn.Module):
    def __init__(self, act="relu"):
        super().__init__()
        self.act = act
        self.conv1 = nn.Conv2d(2, 32, kernel_size = 5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3)
        self.fc1 = nn.Linear(256, 100)
        self.fc2 = nn.Linear(100, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        activation = getActivation(self.act)
        x = activation(F.max_pool2d(self.conv1(x), kernel_size = 2, stride = 2))
        x = self.bn1(x)
        x = activation(F.max_pool2d(self.conv2(x), kernel_size = 2, stride = 1))
        x = self.bn2(x)
        x = activation(self.fc1(x.view(-1, 256)))
        x = activation(self.fc2(x))
        return x
    

# Convolutional Neural Network with auxiliary loss
class CNN_AUX(nn.Module):
    def __init__(self, nb_hidden = 64, act = "relu"):
        super().__init__()
        self.act = act
        self.conv11 = nn.Conv2d(1, 32, kernel_size = 5)
        self.conv12 = nn.Conv2d(32, 64, kernel_size = 3)
        self.fc11 = nn.Linear(256, nb_hidden)
        self.fc12 = nn.Linear(nb_hidden, 10)

        self.conv21 = nn.Conv2d(1, 32, kernel_size = 5)
        self.conv22 = nn.Conv2d(32, 64, kernel_size = 3)
        self.fc21 = nn.Linear(256, nb_hidden)
        self.fc22 = nn.Linear(nb_hidden, 10)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)

        # layer to learn how to compare the two digits
        self.fc_compare = nn.Linear(20, 2)

    def forward(self, xy):
        activation = getActivation(self.act)

        # seperate images
        x = xy.narrow(1, 0, 1)
        y = xy.narrow(1, 1, 1)

        x = activation(F.max_pool2d(self.conv11(x), kernel_size = 2, stride = 2))
        x = self.bn1(x)
        x = activation(F.max_pool2d(self.conv12(x), kernel_size = 2, stride = 1))
        x = self.bn2(x)
        
        y = activation(F.max_pool2d(self.conv21(y), kernel_size = 2, stride = 2))
        y = self.bn1(y)
        y = activation(F.max_pool2d(self.conv22(y), kernel_size = 2, stride = 1))
        y = self.bn2(y)
        
        x = activation(self.fc11(x.view(-1, 256)))
        x = activation(self.fc12(x))

        y = activation(self.fc21(y.view(-1, 256)))
        y = activation(self.fc22(y))

        # contactenate "two images" together
        z = torch.cat((x,y), 1)
        z = self.fc_compare(z)
        return x, y ,z

    
# Siamese convolutional neural networkwith auxiliary loss
class SIAMESE_CNN_AUX(nn.Module):
    def __init__(self, nb_hidden = 64, act = "relu"):
        super().__init__()
        self.act = act
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3)
        self.fc11 = nn.Linear(256, nb_hidden)
        self.fc12 = nn.Linear(nb_hidden, 10)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)

        # layer to learn how to compare the two digits
        self.fc_compare = nn.Linear(20, 2)

    def forward_once(self, x):
        activation = getActivation(self.act)
        x = activation(F.max_pool2d(self.conv1(x), kernel_size = 2, stride = 2))
        x = self.bn1(x)
        x = activation(F.max_pool2d(self.conv2(x), kernel_size = 2, stride = 1))
        x = self.bn2(x)

        x = activation(self.fc11(x.view(-1, 256)))
        x = activation(self.fc12(x))
        return x

    def forward(self, xy):
        # weight sharing between two subnetworks
        x = self.forward_once(xy.narrow(1, 0, 1))
        y = self.forward_once(xy.narrow(1, 1, 1))

        # contactenate "two images" together
        z = torch.cat((x,y), 1)
        z = self.fc_compare(z)
        return x, y ,z

    
# Get desired activation function 
def getActivation(act): 
    if (act == "relu"):
        activation = F.relu
    if (act == "leaky"):
        activation = F.leaky_relu
    if (act == "sigmoid"):
        activation = torch.sigmoid
    return activation
