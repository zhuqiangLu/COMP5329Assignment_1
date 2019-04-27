import numpy as np

class Dropout(object):
    """
    forward propagation: z2 -> a2 -> dropout a2 -> z3
    backward propagationL dz3 -> dropout da2 -> da2 -> dz2
    """
    def __init__(self, drop):

        self.drop = drop
        self.keep = (1-drop)
        self.mask = None


    def drop_forward(self, a, training):
        if(self.keep == 1 or not training):
            return a
        #get the Bernoulli matrix
        ## when I google how to get bernoulli in numpy, it says I can use bonimial
        #then scale it by the keep rate
        self.mask = np.random.binomial(1,self.keep, a.shape)/self.keep

        return a * self.mask


    def drop_backward(self, da):
        if(self.keep == 1):
            return da
        return (da * self.mask) / self.keep
