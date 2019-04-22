import numpy as np
class standard(object):
    def __init__(self, gama = 1, beta = 0):
        self.gama = gama
        self.beta = beta

    def normalize(self, input, epsilon = 1e-8):
        m = input.shape[1]
        #calculate the mean of each features
        mu = np.sum(input, axis = 1, keepdims = True)/m
        #calculate the variance
        theta = np.sum( (input - mu)**2, axis = 1, keepdims = True)/m
        #normalize
        input = (input - mu)/np.sqrt(theta + epsilon)
        #scale and shift
        input = (self.gama*input) + self.beta
        return input

class rescale(object):
    def __init__(self, gama = 1, beta = 0):
        sefl.gama = gama
        self.beta = beta

    def normalize(self, input, epsilon = 1e-8):
        #find the range for each features
        base = np.max(input, axis = 1, keepdims = True) -np.min(input, axis = 1, keepdims = True)
        #normalize
        input = (input  - np.min(input, axis = 1, keepdims = True))/base
        #scale and shift
        input = (self.gama*input) + self.beta
        return input
