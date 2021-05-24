# Authors: Gianni Lodetti, Luca Bracone, Omid Karimi
# Definition of model optimizers 

class SGD():
    '''
    Stochastic gradient descent.
    '''
    def __init__(self, model, lr=0.001):
        if lr < 0.0:
            raise ValueError("negative learning rate".format(lr))
        self.params = model.param()
        self.lr = lr

    def step(self):
        for param in self.params[0]:
            weight, grad = param
            #update weights
            weight.add_(-self.lr * grad)