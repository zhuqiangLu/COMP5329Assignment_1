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


    def forward(self, a, training):
        #get the Bernoulli matrix
        #then scale it by the keep rate
        self.mask = (np.random.binomial(1 ,self.keep, a.shape)/self.keep)
        if(training):
            return a * self.mask
        else:
            return a


    def backward(self, da_drop):
        # a_drop = mask * a
        #d_a_drop/d_a  =  mask
        return (da_drop * self.mask)
