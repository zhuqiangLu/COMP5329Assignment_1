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
        self.mask = np.random.binomial(1,self.keep, a.shape)
        #[True, False] * [10, 1] = [10, 0],
        #scale the output matrix by the probability
        a = (a  * self.mask)/self.keep

        return a

    def drop_backward(self, da, training):
        if(self.keep == 1 or not training):
            return da
        da = (da * self.mask)/self.keep
        return da
