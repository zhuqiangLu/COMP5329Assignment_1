import numpy as np


class tanh(object):

    def forward(self, z):
        return np.tanh(z)

    def backward(self, a, da):
        deri = 1.0 - np.square(a)
        return da * deri

class sigmoid(object):

    def forward(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def backward(self, a, da):
        deri = a * (1 - a )
        return deri * da


class relu(object):

    def forward(self, z):
        mask = z > 0
        return z*mask

    def backward(self, a, da):
        deri = np.int64(a>0)
        return deri * da

class softmax(object):

    def forward(self, z):
        #first calculate the base(the sum of exp(x_i)) for each row
        return np.exp(z)/np.sum(np.exp(z), axis = 0, keepdims = True)
