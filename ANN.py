import numpy as np
import h5py

class Activation(object):

    '''
    Class Activation, accroding to the _init_, the object can be tanh, sigmoid or relu
    Once the object is initialized, the activation function will not be changed
    The object contain the selected activation function and the derivative of that function
    Relu is added by the assignment team

    Arguments:
    x -- the output of a hidden layer before being applied activation function
    a --  the output of a hidden layer
    activation -- String, the selected activation function, tanh by default
    '''
    def __tanh(self, x):
        return np.tanh(x)

    def __tanh_deriv(self, a):
        # a = np.tanh(x)
        return 1.0 - a**2
    def __logistic(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def __logistic_deriv(self, a):
        # a = logistic(x)
        return  a * (1 - a )

    def __relu(self,x):
        # if x > 0 then relu(x) = x, else relu(x) = 0
        mask = x > 0
        return x*mask
    def __relu_deriv(self, a):
        # if x > 0, deriv(relu(x)) = dev(x) = 1,
        # if x<= 0, deriv(relu(x)) = deriv(0) = 0
        return np.int64(a>0)

    def __softmax(self, x):
        #first calculate the base(the sum of exp(x_i)) for each row
        #x is dxn
        base = np.sum(np.exp(x), axis = 0, keepdims = True)
        #use element wise devision to get softmax outcome
        return np.exp(x)/base


    def __init__(self,activation='tanh'):
        if activation == 'logistic':
            self.f = self.__logistic
            self.f_deriv = self.__logistic_deriv
        elif activation == 'tanh':
            self.f = self.__tanh
            self.f_deriv = self.__tanh_deriv
        elif activation == 'relu':
            self.f = self.__relu
            self.f_deriv = self.__relu_deriv
        elif activation == 'softmax':
            self.f = self.__softmax
            self.f_deriv = None



class HiddenLayer(object):

    """
    """
    def __init__(self,n_in, n_out, activation_last_layer='tanh',activation='tanh', W=None, b=None):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_out,n_in)
        and the bias vector b is of shape (n_out, 1).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(W,X) + b)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: string
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input=None
        self.activation=Activation(activation).f

        # activation deriv of last layer
        self.activation_deriv=None
        if activation_last_layer:
            self.activation_deriv=Activation(activation_last_layer).f_deriv


        #initializing the weight and bias term for this layer when it is created
        self.W = np.random.uniform(
                low=-np.sqrt(6. / (n_in + n_out)),
                high=np.sqrt(6. / (n_in + n_out)),
                size=(n_out, n_in)
        )
        if activation == 'logistic':
            self.W *= 4

        self.b = np.zeros((n_out, 1))

        #create instance variables for grad_w and grad_b
        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)

    def forward(self, input, output_layer = False):
        '''
        :type input: numpy.array
        :param input: a symbolic tensor of shape (n_in,n_example)
        :type dropout: double that less or equal to 1
        :param dropout: the drop out probability
        '''
        m = input.shape[1]



        mu = np.sum(input, axis = 1, keepdims = True)/m #calculate the mean of each features
        theta = np.sum( (input - mu)**2, axis = 1, keepdims = True)/m
        input = (input - mu)/np.sqrt(theta + 1e-8)




        lin_output = np.dot(self.W, input) + self.b

        #apply activation funtion
        self.output = (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )

        self.input = input

        return self.output

    def backward(self, delta):
        '''
        :type delta: numpy.array
        :param delta: the derivative of W and b from the last layer, of dimension [d_prev, d_this_layer]
        '''
        m = self.input.shape[1]

        #first calculate the dw for this layer,

        self.grad_W = np.dot(delta, self.input.T)/m
        #db is the sum of row of delta
        self.grad_b = np.sum(delta, axis = 1, keepdims = True)/m

        #print(delta.shape,self.grad_b.shape, self.grad_W.shape)





        if self.activation_deriv:

            da_prev = np.dot(self.W.T, delta)
            da_prev_dz_prev = self.activation_deriv(self.input)


            delta = da_prev * da_prev_dz_prev

        return delta

class MLP:
    """
    This class is the model itself, serves as the abstract facade of the whole neural network
    """

    def __init__(self, layers, activation=[None,'tanh','tanh'], lamda = None):
        """
        :param layers: A list containing the number of units in each layer.
        Should be at least two values
        :param activation: The activation function to be used.
        """

        self.layers=[]
        self.params=[]
        self.activation = activation
        self.lamda = lamda


        ### initialize layers
        for i in range(len(layers)-1):
            #create layers, including output layer accordng to given layer dimension
            self.layers.append(HiddenLayer(layers[i],layers[i+1],activation[i],activation[i+1]))

    def forward(self,input):

        masks = [None] #get the extra mask for the first layer when doing backprop
        for layer in self.layers[:-1]:
            output=layer.forward(input)
            input=output
        #turn of dropout on output layer
        output = self.layers[-1].forward(input, output_layer = True)
        return output


    def criterion_CE(self, y ,y_hat):
        '''
        This method is to calculate the cross entropy of the forwarded result
        (this method is not regularized)

        '''

        # m is the number of samples
        m = y.shape[1]

        #define epsilon just in case y_hat is 0
        epsilon = 1e-8

        #Cross entropy: (-1/m)* Sigma (y_i * log y_hat_i) for all example
        loss = (-1/m) * np.sum(np.sum((y * np.log(y_hat))))
        # dJ/dy_hat = -y/y_hat for every entry
        # dy_hat/dz is y*y matrix
        # dJ/dz = y_hat - y , see report
        # we skip using the derivative of softmax here since it might affects the performance
        delta = y_hat - y



        return loss, delta

    def backward(self,delta):
        delta = self.layers[-1].backward(delta)
        for layer in reversed(self.layers[:-1]):
            delta=layer.backward(delta)


    def update(self,lr):
        for layer in self.layers:

            layer.W -= lr * layer.grad_W
            layer.b -= lr * layer.grad_b

    def fit(self,X,y,learning_rate=0.1, epochs=100,printLoss = True):
        """
        Online learning.
        :param X: Input data or features
        :param y: Input targets
        :param learning_rate: parameters defining the speed of learning
        :param epochs: number of times the dataset is presented to the network for learning
        """
        X=np.array(X)
        y=np.array(y)

        loss = np.zeros(epochs)

        for epoch in range(epochs):
            y_hat= self.forward(X)
            print('epoch: {}, forward done'.format(epoch))
            loss[epoch], delta = self.criterion_CE(y, y_hat)
            print('epoch: {}, ce done'.format(epoch))
            self.backward(delta)
            print('epoch: {}, backpro done'.format(epoch))
            self.update(lr = learning_rate)
            print('epoch: {}, update  done'.format(epoch))
            if printLoss:
                print('epoch: {}, loss: {}'.format(epoch, loss[epoch]))


        return loss

    def predict(self, x):
        x = np.array(x)

        output = nn.forward(x)
        return output

def load_data():

    with h5py.File('Assignment-1-Dataset/train_128.h5','r') as H:
        data = np.copy(H['data'])
    with h5py.File('Assignment-1-Dataset/train_label.h5','r') as H:
        label = np.copy(H['label'])
    with h5py.File('Assignment-1-Dataset/test_128.h5','r') as H:
        test = np.copy(H['data'])
    return data,label,test

def onehot(label):
    return np.eye(np.max(label)+1)[label]

X, label, Y = load_data()
label = onehot(label)

nn = MLP([128,3,10],[None,'tanh','softmax'])
nn.fit(X.T, label.T, learning_rate=10, epochs=10000, printLoss= True)
