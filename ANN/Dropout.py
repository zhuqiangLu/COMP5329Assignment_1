import numpy as np

class Dropout(object):
    """
    forward propagation: z2 -> a2 -> dropout a2 -> z3
    backward propagationL dz3 -> dropout da2 -> da2 -> dz2
    """
    def __init__(self, drop):

        self.drop = drop
        self.keep = (1-drop)
        print(self.drop)
        self.mask = None


    def drop_forward(self, a, training):
        #get the Bernoulli matrix
        ## when I google how to get bernoulli in numpy, it says I can use bonimial
        #then scale it by the keep rate
        self.mask = np.random.binomial(1 ,self.keep, a.shape)/self.keep
        if(training):
            return a * self.mask
        else:
            return a


    def drop_backward(self, da):

        return (da * self.mask)
