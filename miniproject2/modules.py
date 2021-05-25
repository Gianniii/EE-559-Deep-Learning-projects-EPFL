# Authors: Luca Bracone, Gianni Lodetti, Omid Karimi

import torch
import math

from torch import FloatTensor

torch.set_grad_enabled(False)

class Module(object):
    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []

    def zero_grad(self):
        return

#==================================================================================

# Modules for neural networks

# Module for fully-connected layer
class Linear(Module):
    def __init__(self, input_layer_size, output_layer_size, paramInit = "Normal"):
        super().__init__()
        # Weights "w" is a 2d tensor [input_layer_size, output_layer_size]
        # which is the transpose of what might seem "logical"
        # Thanks to broadcasting "w" is going to increase to a 3d tensor when we receive a batch of inputs
        # Bias "b" is a 1d tensor [output_layer_size]
        # We initialize with Xavier method
        #variance = 2.0 / (input_layer_size + output_layer_size)
        #self.w = torch.empty(input_layer_size, output_layer_size).normal_(0, 1)
    
        var = 1 #normal by default
        self.w = torch.empty(input_layer_size, output_layer_size).normal_(0, var) #normal by default
        if paramInit == "He": 
            # 'He initialization' recommends for layers with a ReLU activation
            var = math.sqrt(2/(input_layer_size))
        if paramInit == "Xavier":
            # 'Xavier initialization' recommends for layers with a tanh activation
            var =  math.sqrt(2/(input_layer_size + output_layer_size))
            
        self.w = torch.empty(input_layer_size, output_layer_size).normal_(0, var)
        
        self.b = torch.empty(output_layer_size).normal_(0, var)
        # Gradient vector is just empty for now
        # Each channel represents one of the inputs we receive in the batch
        # And within each channel, each entry represents "how much" the weight should change according to that x
        self.grad_w = torch.empty(self.w.size()).fill_(0)
        self.grad_b = torch.empty(self.b.size()).fill_(0)


    def forward(self, x):
        # We record the input for later use
        self.input = x
        # It's just a matrix-vector product plus bias after it
        return (x @ self.w) + self.b

    def backward(self, dl_dout):
        #print(dl_dout.shape)
        #print(self.input.shape)
        #print(self.grad_w.shape)
        #print("gradients: \n")
        #print(self.grad_w)
        self.grad_w.add_(self.input.t() @ dl_dout)
        self.grad_b.add_(dl_dout.sum(0))
        #print(self.grad_w)
        return dl_dout @ self.w.t()

    def param(self):

        return [
                (self.w, self.grad_w),
                (self.b, self.grad_b)
                ]

    def zero_grad(self):
        self.grad_w.zero_()
        self.grad_b.zero_()



# Module to combines several other modules
class Sequential(Module):
    def __init__(self, modules):
        super().__init__()
        self.modules = modules

    def forward(self, x):
        self.x = x
        for m in self.modules:
            x = m.forward(x)
        return x

    def backward(self, dl_dout):
        self.dl_dout = dl_dout
        for m in reversed(self.modules):
            dl_dout = m.backward(dl_dout)
        return dl_dout

    def param(self):
        param = []
        for m in self.modules:
            param.extend(m.param())
        return param
    
    def zero_grad(self):
        for m in self.modules:
            m.zero_grad()

#==================================================================================

# Modules for activations functions

class ReLU(Module):
    def forward(self, x):
        self.x = x
        return torch.max(torch.zeros_like(x), x)

    def backward(self, dl_dout):
        # clamp forces negative elements to 0.0
        return torch.clamp(self.x.sign(), 0.0, 1.0) * dl_dout

class Tanh(Module):
    def forward(self, x):
        self.x = x
        return x.tanh()
    
    def backward(self, dl_dout):
        return 4.0 * (self.x.exp() + (-self.x).exp()).pow(-2) * dl_dout

class Sigmoid(Module):
    def forward(self, x):
        self.x = x.clone()
        return torch.div(1, (1+ torch.exp(-self.x)))

    def backward(self, dl_dout):
        sig = torch.div(1, (1+ torch.exp(-self.x)))
        return sig * (1-sig) * dl_dout
#==================================================================================

# Modules for loss functions

class MSELoss(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input, target):
        self.input = input
        self.target = target.view(input.shape)
        loss = (self.target - input).pow(2)
        return torch.mean(loss, 1).view(input.shape)

    def backward(self):
        return torch.div(self.input - self.target, self.input.size(0)) * 2

#class CrossEntropyLoss(Module):
    #def __init__(self) -> None:
    #    super().__init__()

    #def forward(self, input, target):
    #    sig = input
    #    self.p = sig
    #    self.y = target.view(input.shape)
    #    loss = self.y*(self.p.log()) - (1-self.y) * (1 -self.p)
    #    return loss

    #def backward(self):
    #    return ((-self.y)/self.p) + (1-self.y)/(1-self.p)
    
