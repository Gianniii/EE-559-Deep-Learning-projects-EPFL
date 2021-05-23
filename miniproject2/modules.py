# Authors: Luca Bracone, Gianni Lodetti, Omid Karimi

import torch
import math

torch.set_grad_enabled(False)

class Module(object):
    def zero_grad(self):
        """
        Resets the gradients to zero
        Do not call this on a "Module" with no grad_w or grad_b !!!
        """
        self.grad_w = tensor.empty(self.w.size()).fill_(0)
        self.grad_b = tensor.empty(self.b.size()).fill_(0)

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
    def __init__(self, input_layer_size, output_layer_size):
        # Weights "w" is a 2d tensor [input_layer_size, output_layer_size]
        # which is the transpose of what might seem "logical"
        # Thanks to broadcasting "w" is going to increase to a 3d tensor when we receive a batch of inputs
        # Bias "b" is a 1d tensor [output_layer_size]
        # We initialize with Xavier method
        variance = 2.0/(input_layer_size + output_layer_size)
        self.w = tensor.empty(input_layer_size, output_layer_size).normal_(0, variance)
        self.b = tensor.empty(output_layer_size).normal_(0, variance)

        # Gradient vector is just empty for now
        # Each channel represents one of the inputs we receive in the batch
        # And within each channel, each entry represents "how much" the weight should change according to that x
        self.grad_w = tensor.empty(self.w.size()).fill_(0)
        self.grad_b = tensor.empty(self.b.size()).fill_(0)

    def forward(self, x):
        # We record the input for later use
        self.input = x
        # It's just a matrix-vector product plus bias after it
        return (self.w.t() @ x) + self.b

    def backward(self, dl_dout):
        self.grad_w.add_(dl_dout.t() @ self.input)
        self.grad_b.add_(dl_dout.t())

    def param(self):

        return [
                (self.w, self.grad_w),
                (self.b, self.grad_b)
                ]


# Module to combines several other modules
class Sequential(Module):
    def __init__(self, modules):
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
        return dl_out

    def param(self):
        param = []
        for m in self.modules:
            param.append(m.param())
        return param

#==================================================================================

# Modules for activations functions

class ReLU(Module):
    def __init__(self):

    def forward(self):

    def backward(self):

    def param(self):



class Tanh(Module):
    def forward(self, x):
        self.x = x
        return x.tanh()
    def backward(self, dl_dout):
        return 4.0 * (self.x.exp() + (- self.x).exp()).pow(-2) * dl_dout

#==================================================================================

# Modules for loss functions

class MSELoss(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input, target):
        self.input = input
        self.target = target
        loss = (target-input).pow(2)
        return torch.mean(loss)

    def backward(self):
        return 2 * (self.input - self.target).div(self.input.size(0))
