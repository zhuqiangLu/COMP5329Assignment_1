from Activation import tanh, sigmoid, softmax, relu
from BatchNormalizer import standard, rescale
from Layer import HiddenLayer
from Initializer import Xavier, He, dumbInitializer
from Cost import Cross_Entropy
from Dropout import Dropout
from Momentum import Standard_Momentum, Nesterov
from Regularizer import L1, L2, dumbRegularizer
from SGD import Batch


import numpy as np
import h5py
import matplotlib.pyplot as plt



class Model(object):

    def __init__(self,
                training_data,
                training_label,
                learning_rate = 0.01,
                batch_size = None,
                drop = 0,
                momentum = None,
                cost = Cross_Entropy(),
                regularizer = dumbRegularizer()):

        self.batch = Batch(training_data, training_label)
        self.batch_size = batch_size
        self.dev_X = None
        self.dev_Y = None
        self.classes = training_label.shape[0]
        self.lr = learning_rate
        self.dims = [training_data.shape[0]]
        self.layers = []
        self.drop = drop
        self.momentum = momentum
        self.cost = cost
        self.batch_size = batch_size
        self.regularizer = regularizer
        self.printInfo = False
        self.printAt = 1
        self.plot = []

    def add_layer(self,
                n_out,
                ini = Xavier(),
                acti = tanh(),
                norm = standard(),
                drop = None):

        if(drop == None):
            drop = self.drop
        n_in = self.dims[-1]
        layer = HiddenLayer(n_in, n_out, ini)
        layer.setActivation(acti)
        layer.setBatchNormalizer(norm)
        layer.setDropout(drop = drop)

        if(self.momentum != None):
            layer.setMomentum(self.momentum.clone())

        self.dims.append(n_out)
        self.layers.append(layer)

    def add_last_layer(self,
                    ini = Xavier(),
                    acti = softmax(),
                    norm = standard()):

        n_in = self.dims[-1]
        n_out = self.classes
        layer = HiddenLayer(n_in, n_out, ini, last_layer = True)
        layer.setActivation(acti)
        layer.setBatchNormalizer(norm)
        #last layer dose not need Dropout
        layer.setDropout(drop = 0)

        if(self.momentum != None):
            layer.setMomentum(self.momentum.clone())

        self.layers.append(layer)

    def set_dev(self, dev_X, dev_Y):
        self.dev_X = dev_X
        self.dev_Y = dev_Y


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

        loss_train = []
        loss_dev = []
        #first get batch
        for i in range(epoch):
            self.batch.fit(self, size = self.batch_size)
            if(self.printInfo):
                print("epoch {}, loss {}, accuracy on training data {}".format(i, self.batch.getLoss(), self.batch.getAccuracy()))
                loss_train.append(self.batch.getLoss())
                if(self.dev_X is not None and self.dev_Y is not None):
                    pred_dev = self.predict(self.dev_X)
                    dev_accu = np.mean( np.equal(np.argmax(self.dev_Y, 0), np.argmax(pred_dev, 0)))
                    loss_dev.append(self.cost.loss(dev_Y, pred_dev))
                    print("dev accuracy {}".format(dev_accu))

        self.plot.append(loss_train)
        self.plot.append(loss_dev)




    def plotLoss(self):
        plt.plot(np.arange(50), self.plot[0], label = "train_loss")
        plt.plot(np.arange(50), self.plot[1], label = "dev_loss")

        plt.xlabel("epoch")
        plt.ylabel("loss")

        plt.legend()
        plt.title("loss")

        plt.show()


    def predict(self, x):
        x = np.array(x)
        for layer in self.layers:
            x = layer.forward(x, training = False)#regularizer collect W during forward
        return x

    def test(self, test_x, test_y):
        pred_test = self.predict(test_x)
        test_accu = np.mean( np.equal(np.argmax(test_y, 0), np.argmax(pred_test, 0)))
        print("test accuracy: {}".format(test_accu))





def load_data():

    with h5py.File('Assignment-1-Dataset/train_128.h5','r') as H:
        data = np.copy(H['data'])
    with h5py.File('Assignment-1-Dataset/train_label.h5','r') as H:
        label = np.copy(H['label'])
    with h5py.File('Assignment-1-Dataset/test_128.h5','r') as H:
        test = np.copy(H['data'])
    return data,label,test

def train_dev_test(X, Y, train_rate, dev_rate, test_rate):
    m = X.shape[1]
    train = round(m*train_rate)
    dev = train + round(m*dev_rate)
    test = dev + round(m*test_rate)
    return [(X[:,:train], Y[:,:train]),(X[:,train:dev], Y[:, train:dev]),(X[:, dev:], Y[:, dev:])]


def onehot(label):
    return np.eye(np.max(label)+1)[label]

if __name__ == "__main__":
    X, label, toPredict= load_data() # X is(m, n_in)
    label = onehot(label)

    X = X.T
    Y = label.T
    toPredict = toPredict.T

    data = train_dev_test(X, Y, 0.8, 0.1, 0.1)
    (train_X, train_Y) = data[0]
    print(train_X.shape, train_Y.shape)
    (dev_X, dev_Y) = data[1]
    print(dev_X.shape, dev_Y.shape)
    (test_X, test_Y) = data[2]

    model = Model(train_X, train_Y, batch_size = 30, drop = 0.1, regularizer = L2(0.01),momentum = Nesterov())
    model.print_Info(True, 1)
    model.set_dev(dev_X, dev_Y)
    model.add_layer(192, ini = He(), acti = relu())
    model.add_layer(92, ini = He(), acti = relu())
    model.add_layer(48, ini = He(), acti = relu())
    model.add_last_layer()
    model.fit(epoch = 50, lr = 0.005)
    model.plot()
    model.test(test_X, test_Y)
