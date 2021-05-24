# Authors: Gianni Lodetti, Luca Bracone, Omid Karimi
# Definition of model optimizers 

class SGD():
    '''
    Stochastic gradient descent.
    '''
    def __init__(self, param, lr=0.01):
        if lr < 0.0:
            raise ValueError("negative learning rate".format(lr))
        self.params = param
        self.lr = lr

    def step(self):
        for param in self.params[0]:
            weight, grad = param
            #update weights
            print("==============================\n")
            print(weight)
            print(grad)
            weight.add_(-self.lr * grad)
            print(weight)