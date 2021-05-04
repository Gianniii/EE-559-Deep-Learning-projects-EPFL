import torch
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
        x = F.relu(self.fc1(x.view(-1, 256)))
        #print("d2", x.size())
        x = F.relu(self.fc2(x))
        #print("e", x.size())
        return x


#THIS MODEL TAKES MORE THEN 2 SECONDS TO TRAIN!! NEED TO FIND WAYS TO SPEED IT UP!!!
#TODO try leakyrelu, sigmoid, adding, add weight sharing through siamese network, playing with kernel sizes ect.. and different optimizer functions too
#add auxiliary loss we need to distinguish the images in the image paire(to take advantage of the classes)
class CNN_AUX(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv11 = nn.Conv2d(1, 32, kernel_size = 5)
        self.conv12 = nn.Conv2d(32, 64, kernel_size = 3)
        self.fc11 = nn.Linear(256, 100)
        self.fc12 = nn.Linear(100, 10)

        self.conv21 = nn.Conv2d(1, 32, kernel_size = 5)
        self.conv22 = nn.Conv2d(32, 64, kernel_size = 3)
        self.fc21 = nn.Linear(256, 100)
        self.fc22 = nn.Linear(100, 10)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)

        #layer to learn how to compare the two digits
        self.fc_compare = nn.Linear(20, 2)


    def forward(self, xy):
        #seperate images
        x = xy.narrow(1, 0, 1)
        y = xy.narrow(1, 1, 1)


        x = F.relu(F.max_pool2d(self.conv11(x), kernel_size = 2, stride = 2))
        x = self.bn1(x)
        x = F.relu(F.max_pool2d(self.conv12(x), kernel_size = 2, stride = 1))
        x = self.bn2(x)
        
        y = F.relu(F.max_pool2d(self.conv21(y), kernel_size = 2, stride = 2))
        y = self.bn1(y)
        y = F.relu(F.max_pool2d(self.conv22(y), kernel_size = 2, stride = 1))
        y = self.bn2(y)
        
        x = F.relu(self.fc11(x.view(-1, 256)))
        x = F.relu(self.fc12(x))

        y = F.relu(self.fc21(y.view(-1, 256)))
        y = F.relu(self.fc22(y))

        #contactenate "two images" together
        z = torch.cat((x,y), 1)
        z = self.fc_compare(z)
        return x, y ,z