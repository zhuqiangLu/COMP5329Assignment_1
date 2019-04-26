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
        self.n_in = n_in
        self.n_out = n_out
        self.activation = Activation.tanh()
        self.BatchNormalizer = norm.standard()
        self.optimizer = None
        self.m = None
        self.z = None
        self.a = None
        self.a_drop = None
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

    def setOptimizer(self, optimizer):
        if(optimizer != None):
            self.optimizer = optimizer




    def forward(self,input, training = True, regularizer = None):
        self.m = input.shape[1]

        assert(self.n_in == input.shape[0])

        #let regularizer collect W
        if(regularizer is not None):

            regularizer.collectW(self.layer.W)

        #-> input ->  norm_input -> z1 -> a1   ->  a1_drop ->

        input = self.BatchNormalizer.forward(input)
        self.input = input
        self.z = self.Layer.forward(input)
        self.a = self.activation.activate(self.z)
        self.a_drop = self.dropout.drop_forward(self.a, training)
        return self.a

    def backward(self, da):
        #da means dj/da_drop here
        # <- dinput <- dinput_norm <- dz1 <- da1  <- da1_drop <-

        #first get da_dz of the activation function of this layer
        #if this is the last layer, dz is given, but not da, therefore we skip calculating da_dz
        da_dz = np.array(1)

        if(not self.last_layer):

            da = self.dropout.drop_backward(da)
            da_dz = self.activation.derivative(self.a)

        dz= da * da_dz
        #then backward the norm layer
        #then update dw and db using dz, calculate dj/da_drop,
        din_norm = self.Layer.backward(dz)

        din = self.BatchNormalizer.backward(din_norm)
        #then return dj_dz
        return din

    def update(self,lr=0.01, regularizer = None):
        #self.Layer.grad_W = regularizer.update_grad_W(self.Layer.grad_W,self.Layer.W, self.m)

        if(self.optimizer != None):
            self.Layer.W = self.optimizer.update_W(lr, self.Layer.W, self.Layer.grad_W)
            self.Layer.b = self.optimizer.update_b(lr, self.Layer.b, self.Layer.grad_b)
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
        :param n_in: dim of input

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

        self.grad_b = np.sum(dz, axis = 1, keepdims = True)
        self.grad_b = np.mean(self.grad_b, axis = 1, keepdims = True)


        #calculate da of this layers
        da = np.dot(self.W.T, dz)

        return da
