from Activation import tanh, sigmoid, softmax, relu
from BatchNormalizer import standard, rescale
from Layer import HiddenLayer
from Initializer import Xavier, He, dumbInitializer
from Cost import Cross_Entropy
from Dropout import Dropout
from Regularizer import L1, L2, dumbRegularizer
import numpy as np
import h5py


class Model(object):

    def __init__(self, training_data, training_label, cost = Cross_Entropy(), regularizer = dumbRegularizer() , lamda = 1):
        self.X = training_data
        self.Y = training_label
        self.dims = [training_data.shape[0]]
        self.layers = []
        self.cost = cost
        self.regularizer = regularizer
        self.regularizer.setLamda(lamda)
        self.printloss = False
        self.printAt = 1

    def add_layer(self, n_out, ini = Xavier(), acti = tanh(), norm = standard(), drop = 0):
        n_in = self.dims[-1]
        layer = HiddenLayer(n_in, n_out, ini)
        layer.setActivation(acti)
        layer.setBatchNormalizer(norm)
        layer.setDropout(drop = drop)
        self.dims.append(n_out)
        self.layers.append(layer)

    def add_last_layer(self,ini = Xavier(), acti = softmax(), norm = standard()):
        n_in = self.dims[-1]
        n_out = self.Y.shape[0]
        layer = HiddenLayer(n_in, n_out, ini, last_layer = True)
        layer.setActivation(acti)
        layer.setBatchNormalizer(norm)
        #last layer dose not need Dropout
        layer.setDropout(drop = 0)
        self.layers.append(layer)

    def printLoss(self,printloss = False, printAt = 1):
        self.printloss = printloss
        self.printAt = printAt

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, dz):
        da = dz
        for layer in reversed(self.layers):
            da = layer.backward(da)

    def update(self, lr):
        for layer in self.layers:
            layer.update(lr, self.regularizer)

    def fit(self, epoch = 500, lr = 0.01):

        for i in range(epoch):
            input = self.X
            y_hat = self.forward(input)
            loss = self.cost.loss(self.Y, y_hat) + self.regularizer.loss
            dz = self.cost.dz(self.Y, y_hat)
            self.backward(dz)
            self.update(lr)
            if(self.printLoss and (i % self.printAt) == 0):
                print(i, loss)

    def predict(self, x):
        x = np.array(x)
        return self.forward(x)




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

if __name__ == "__main__":
    X, label, Y = load_data()
    label = onehot(label)
    model = Model(X.T, label.T, regularizer = L2(), lamda = 0.9 )
    model.add_layer(64)
    model.add_layer(32)
    model.add_last_layer()
    model.printLoss(printloss = True, printAt = 1)
    model.fit(epoch = 10000, lr = 0.03)
