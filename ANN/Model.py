from Activation import tanh, sigmoid, softmax, relu
from BatchNormalizer import standard, rescale
from Layer import HiddenLayer
from Initializer import Xavier, He, dumbInitializer
from Cost import Cross_Entropy
from Dropout import Dropout
from Regularizer import L1, L2, dumbRegularizer
from SGD import Batch
import numpy as np
import h5py



class Model(object):

    def __init__(self, training_data, training_label, learning_rate = 0.01, batch_size = 1, cost = Cross_Entropy(), regularizer = dumbRegularizer() , lamda = 1, drop = 0):

        self.batch = Batch(training_data, training_label)

        self.classes = training_label.shape[0]
        self.lr = learning_rate
        self.dims = [training_data.shape[0]]
        self.layers = []
        self.drop = drop
        self.cost = cost
        self.batch_size = batch_size
        self.regularizer = regularizer
        self.regularizer.setLamda(lamda)
        self.printInfo = False
        self.printAt = 1

    def add_layer(self, n_out, ini = Xavier(), acti = tanh(), norm = standard(), drop = None):
        if(drop == None):
            drop = self.drop
        n_in = self.dims[-1]
        layer = HiddenLayer(n_in, n_out, ini)
        layer.setActivation(acti)
        layer.setBatchNormalizer(norm)
        layer.setDropout(drop = drop)
        self.dims.append(n_out)
        self.layers.append(layer)

    def add_last_layer(self,ini = Xavier(), acti = softmax(), norm = standard()):
        n_in = self.dims[-1]
        n_out = self.classes
        layer = HiddenLayer(n_in, n_out, ini, last_layer = True)
        layer.setActivation(acti)
        layer.setBatchNormalizer(norm)
        #last layer dose not need Dropout
        layer.setDropout(drop = 0)
        self.layers.append(layer)

    def print_Info(self,printInfo = False, printAt = 1):
        self.printInfo = printInfo
        self.printAt = printAt

    def forward(self, input):
        self.regularizer.reset()#reset the W stored in regularizer
        for layer in self.layers:
            input = layer.forward(input, self.regularizer)#regularizer collect W during forward
        return input

    def backward(self, dz):
        da = dz
        for layer in reversed(self.layers):
            da = layer.backward(da)

    def update(self):
        for layer in self.layers:
            layer.update(self.lr, self.regularizer)


    def fit(self, epoch = 100, lr = 0.01):

        #first get batch
        for i in range(epoch):
            self.batch.fit(self, size = 30)
            if(self.printInfo):
                print("epoch {}, loss {}, accuracy on training data {}".format(i, self.batch.getLoss(), self.batch.getAccuracy()))



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
    model = Model(X.T, label.T, batch_size = 30, drop = 0.2)
    model.add_layer(192, ini = He(), acti = relu())
    model.add_layer(92, ini = He(), acti = relu())
    model.add_layer(48, ini = He(), acti = relu())
    model.add_last_layer()
    model.print_Info(True, 1)
    model.fit(epoch = 1000, lr = 0.005)
