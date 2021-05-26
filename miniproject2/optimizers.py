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
        for param in self.params:
            weight, grad = param
<<<<<<< HEAD
            #update weights
            weight.add_(-self.lr * grad)
            
=======
            # update weights
            weight.add_(-self.lr * grad)
        
>>>>>>> 2cceca5d0494c94c5ce4d067528bd1ef1b314f4b
