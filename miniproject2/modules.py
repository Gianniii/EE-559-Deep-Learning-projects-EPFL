# Authors: Luca Bracone, Gianni Lodetti, Omid Karimi

import torch
import math

torch.set_grad_enabled(False)

class Module(object):
    
    def forward(self, *input):
        raise NotImplementedError
        
    def backward(self, *gradwrtoutput):
        raise NotImplementedError
        
    def param(self):
        return []
     
#==================================================================================        
  
# Modules for neural networks    
    
# Module for fully-connected layer
class Linear(Module):
    def __init__(self):
        
    def forward(self):
        
    def backward(self):
        
    def param(self):
        
    
# Module to combines several other modules
class Sequential(Module):
    def __init__(self):
        
    def forward(self):
        
    def backward(self):
        
    def param(self):
        

#==================================================================================

# Modules for activations functions

class ReLU(Module):
    def __init__(self):
        
    def forward(self):
        
    def backward(self):
        
    def param(self): 
        
        
        
class Tanh(Module):
    def __init__(self):
        
    def forward(self):
        
    def backward(self):
        
    def param(self):        



#==================================================================================

# Modules for loss functions

class MSELoss(Module):
    def __init__(self):
        
    def forward(self):
        
    def backward(self):
        
    def param(self): 
        
        