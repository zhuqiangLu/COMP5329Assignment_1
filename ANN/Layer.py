import numpy as np
import Initializer as ini
import BatchNormalizer as norm
from Activation import *
from Dropout import *
from Regularizer import *

class HiddenLayer(object):
    '''
        the controller object that wrap layer with activation function, optimization function, initializer, normalizer.
        This object also control forward and backward propagation
    '''
    def __init__(self, n_in, n_out, ini, last_layer = False):
        self.FC = FullyConnected(n_in, n_out, ini)
        self.n_in = n_in
        self.n_out = n_out
        self.activation = tanh()
        self.BatchNormalizer = None
        self.optimizer = None
        self.m = None
        self.z = None
        self.z_norm = None
        self.a = None
        self.a_drop = None
        self.drop = None
        self.input = None
        self.last_layer = last_layer

    def setDropout(self, drop):
        self.drop = Dropout(drop)

    def setActivation(self,activation):
        if(activation != None):
            self.activation = activation

    def setBatchNormalizer(self,norm):
        if( norm != None):
            self.BatchNormalizer = norm

    def setOptimizer(self, optimizer):
        if(optimizer != None):
            self.optimizer = optimizer




    def forward(self, input, training = True, regularizer = None):
        self.m = input.shape[1]
        self.input = input
        assert(self.n_in == input.shape[0])

        #-> input ->  z1 -> norm_z1 -> a1   ->  a1_drop ->

        #-> FC -> Batch Norm -> Activation -> Drop out ->FC ..
        if(self.optimizer.__class__.__name__ == "Nesterov"):
            last_v_W = self.optimizer.get_last_v_W()
            last_v_b = self.optimizer.get_last_v_b()
            gama = self.optimizer.gama
            self.FC.W -= gama*last_v_W
            self.FC.b -= gama*last_v_b

        if(not training):
            regularizer = None

        self.z = self.FC.forward(input, regularizer)

        if(self.BatchNormalizer is not None):
            self.z_norm = self.BatchNormalizer.forward(self.z, training)
        else:
            self.z_norm = self.z

        self.a = self.activation.forward(self.z_norm)
        self.a_drop = self.drop.forward(self.a, training)
        return self.a_drop




    def backward(self, da_drop, regularizer = None):
        #da means dj/da_drop here
        # <- dinput <- dz1 <- dz1_norm <- da1  <- da1_drop <-

        #first get da_dz of the activation function of this layer
        #if this is the last layer, dz is given, but not da, therefore we skip calculating da_dz

        dz = None
        if(self.last_layer):
            dz = da_drop
        else:
            da = self.drop.backward(da_drop)
            if(self.activation is not None):
                dz_norm = self.activation.backward(self.a, da)
            else:
                dz_norm = da
            if(self.BatchNormalizer is not None):
                dz = self.BatchNormalizer.backward(dz_norm)
            else:
                dz = dz_norm

        dinput = self.FC.backward(dz, regularizer)
        return dinput


    def update(self,lr):

        if(self.optimizer != None):
            self.FC.W = self.optimizer.update_W(lr, self.FC.W, self.FC.grad_W)
            self.FC.b = self.optimizer.update_b(lr, self.FC.b, self.FC.grad_b)
        else:
            self.FC.W = self.FC.W - lr * self.FC.grad_W
            self.FC.b = self.FC.b - lr * self.FC.grad_b

        #update normalizer as well
        if(self.BatchNormalizer is not None):
            self.BatchNormalizer.update(lr)



class FullyConnected(object):


    def __init__(self, n_in, n_out, ini):

        self.input = None
        #init W and b using the given initializer
        self.W = ini.get_W(n_in, n_out)
        self.b = ini.get_b(n_out)

        #create instance variables for grad_w and grad_b
        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)

    def forward(self, input, regularizer = None):

        if regularizer is not None:
            regularizer.forward(self.W)
        self.input = input
        return np.dot(self.W,input) + self.b


    def backward(self, dz, regularizer = None):


        m = self.input.shape[1]

        #first calculate the dw for this layer,
        #dw = dj/dz * dz/dw <- the input of this layer
        self.grad_W = np.dot(dz, self.input.T)/m
        if(regularizer is not None):
            self.grad_W = regularizer.backward(self.grad_W, self.W, m)

        #db is the sum of row of delta

        self.grad_b = np.sum(dz, axis = 1, keepdims = True)/m


        #calculate da of this layers
        da = np.dot(self.W.T, dz)

        return da
