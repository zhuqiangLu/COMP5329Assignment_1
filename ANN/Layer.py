import numpy as np
import Initializer as ini
import BatchNormalizer as norm
import Activation
from Dropout import Dropout
from Regularizer import L1, L2, dumbRegularizer

class HiddenLayer(object):
    '''
        the controller object that wrap layer with activation function, optimization function, initializer, normalizer.
        This object also control forward and backward propagation
    '''
    def __init__(self, n_in, n_out, ini, last_layer = False):
        self.Layer = Layer(n_in, n_out, ini)
        self.activation = Activation.tanh()
        self.BatchNormalizer = norm.standard()
        self.momentum = None
        self.m = None
        self.z = None
        self.a = None
        self.dropout = None
        self.input = None
        self.last_layer = last_layer

    def setDropout(self, drop):
        self.dropout = Dropout(drop)

    def setActivation(self,activation):
        if(activation != None):
            self.activation = activation

    def setBatchNormalizer(self,norm):
        if( norm != None):
            self.BatchNormalizer = norm

    def setMomentum(self, momentum):
        if(momentum != None):
            self.momentum = momentum


    def forward(self,input, training = True, regularizer = None):
        self.m = input.shape[1]
        #let regularizer collect W
        if(regularizer is not None):
            regularizer.collectW(self.layer.W)

        #z1   ->     a1   ->  a1_drop  ->  z2
        input = self.BatchNormalizer.normalize(input)
        self.input = input
        self.z = self.Layer.forward(input)
        self.a = self.activation.activate(self.z)
        self.a = self.dropout.drop_forward(self.a, training)
        return self.a

    def backward(self, da, training = True):
        #da means dj/da_drop here
        #z1 <-  a1  <- a1_drop <- z2

        #first get da_dz of the activation function of this layer
        #if this is the last layer, dz is given, but not da, therefore we skip calculating da_dz
        da_dz = np.array(1)

        if(not self.last_layer):

            da = self.dropout.drop_backward(da, training)
            da_dz = self.activation.derivative(self.a)

        dz = da * da_dz
        #then update dw and db using dz, calculate dj/da_drop,
        dj_da = self.Layer.backward(dz)


        #then return dj_dz
        return dj_da

    def update(self,lr=0.01, regularizer = None):
        self.Layer.grad_W = regularizer.update_grad_W(self.Layer.grad_W,self.Layer.W, self.m)

        if(self.momentum != None):
            self.Layer.W = self.momentum.update_W(lr, self.Layer.W, self.Layer.grad_W)
            self.Layer.b = self.momentum.update_b(lr, self.Layer.b, self.Layer.grad_b)
        else:
            self.Layer.W -= lr * self.Layer.grad_W
            self.Layer.b -= lr * self.Layer.grad_b



class Layer(object):
    '''
    a layer without activation fuction, optimization
    '''

    def __init__(self, n_in, n_out, ini):
        """
        Hidden unit activation is given by: tanh(dot(W,X) + b)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type ini: Initializer object
        :param ini: initialization method of W


        """
        self.input = None
        #init W and b using the given initializer
        self.W = ini.get_W(n_in, n_out)
        self.b = ini.get_b(n_out)
        #create instance variables for grad_w and grad_b
        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)

    def forward(self, input):
        """
        Forward propagation of a hidden layer

        :type input: matrix that of the shape (n_in, n_example)
        :param input: the input matrtix

        """
        self.input = input

        return np.dot(self.W,input) + self.b


    def backward(self, dz):
        '''
        :type delta: numpy.array
        :param delta: the derivative of W and b from the last layer, of dimension [d_prev, d_this_layer]
        '''

        m = self.input.shape[1]

        #first calculate the dw for this layer,
        #dw = dj/dz * dz/dw <- the input of this layer
        self.grad_W = np.dot(dz, self.input.T)/m
        #db is the sum of row of delta
        self.grad_b = np.sum(dz, axis = 1, keepdims = True)/m

        #calculate da of this layers
        da = np.dot(self.W.T, dz)

        return da
