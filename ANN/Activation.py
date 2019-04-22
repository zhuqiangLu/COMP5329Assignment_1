import numpy as np


class tanh(object):

    def activate(self, z):
        return np.tanh(z)

    def derivative(self, a):
        return 1.0 - a**2

class sigmoid(object):

    def activate(self, z):
        return 1.0 / (1.0 + np.exp(-x))

    def derivative(self, a):
        return  a * (1 - a )

class relu(object):

    def activate(self, z):
        # if x > 0 then relu(x) = x, else relu(x) = 0
        mask = x > 0
        return x*mask

    def derivative(self, a):
        # if x > 0, deriv(relu(x)) = dev(x) = 1,
        # if x<= 0, deriv(relu(x)) = deriv(0) = 0
        return np.int64(a>0)

class relu(object):

    def activate(self, z):
        mask = x > 0
        return x*mask

    def derivative(self, a):
        return np.int64(a>0)

class softmax(object):

    def activate(self, z):
        #first calculate the base(the sum of exp(x_i)) for each row
        #x is dxn
        base = np.sum(np.exp(z), axis = 0, keepdims = True)
        #use element wise devision to get softmax outcome
        return np.exp(z)/base

    def derivative(self,a):
        #not recommanded to use
        m = a.shape[1]
        f = a.shape[0]
        #first, create spaceholder for the matrix, the dy/dz matrix will be (y,y,n)
        deri = np.zeros((f,f,m))

        #then create the mask for Kronecker delta
        K = np.eye(f)

        for n in range(m):
            y_hat = a[:,n]
            deri[:,:,n] = (K - y_hat)*y_hat.reshape(f,1)

        return deri
