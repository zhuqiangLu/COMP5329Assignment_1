from Activation import tanh, sigmoid, softmax, relu
from BatchNormalizer import standard
from Layer import HiddenLayer
from Initializer import Xavier, He
from Cost import Cross_Entropy
from Dropout import Dropout
from Optimizer import Momentum, Nesterov, AdaGrad, AdaDelta, Adam
from Regularizer import L1, L2
from SGD import Batch


import numpy as np
import h5py
import matplotlib.pyplot as plt



class Model(object):

    def __init__(self,
                training_data,
                training_label,
                learning_rate = 0.0001,
                batch_size = None,
                drop = 0,
                optimizer = None,
                norm = None,
                cost = Cross_Entropy(),
                regularizer = None):

        self.batch = Batch(training_data, training_label)
        self.X = training_data
        self.Y = training_label
        self.batch_size = batch_size
        self.m = training_data.shape[1]
        self.dev_X = None
        self.dev_Y = None
        self.classes = training_label.shape[0]
        self.dims = [training_data.shape[0]]
        self.lr = learning_rate
        self.layers = []
        self.drop = drop
        self.optimizer = optimizer
        self.norm = norm

        if(self.optimizer is not None and self.norm is not None):
            self.norm.set_optimizer(self.optimizer.clone())

        self.cost = cost
        self.regularizer = regularizer
        self.printInfo = False
        self.printAt = 1
        self.plot = []

    def add_layer(self,
                n_out,
                ini = Xavier(),
                acti = relu(),
                drop = None):

        drop = self.drop
        n_in = self.dims[-1]
        layer = HiddenLayer(n_in, n_out, ini)
        layer.setActivation(acti)

        if(self.norm is not None):
            layer.setBatchNormalizer(self.norm.clone())

        layer.setDropout(drop = drop)

        if(self.optimizer != None):
            layer.setOptimizer(self.optimizer.clone())

        self.dims.append(n_out)
        self.layers.append(layer)


    def add_last_layer(self,
                    ini = Xavier(),
                    acti = softmax()):

        n_in = self.dims[-1]
        n_out = self.classes
        layer = HiddenLayer(n_in, n_out, ini, last_layer = True)
        layer.setActivation(softmax())

        #last layer dose not need Dropout
        layer.setDropout(drop = 0)
        if(self.optimizer != None):
            layer.setOptimizer(self.optimizer.clone())

        self.layers.append(layer)

    def set_dev(self, dev_X, dev_Y):
        self.dev_X = dev_X
        self.dev_Y = dev_Y

    def get_reg_loss(self):
        if(self.regularizer is None):
            return 0
        else:
            return self.regularizer.get_loss(self.m)

    def reset_regularizer(self):
        if(self.regularizer is not None):
            self.regularizer.reset()

    def print_Info(self,printInfo = False, printAt = 1):
        self.printInfo = printInfo
        self.printAt = printAt

    def forward(self, input, training = True):

        #reset regularizer before each forward pass
        self.reset_regularizer()
        for layer in self.layers:
            input = layer.forward(input, training = training, regularizer = self.regularizer)#regularizer collect W during forward

        return input

    def backward(self, dz):
        da = dz
        for layer in reversed(self.layers):
            da = layer.backward(da, self.regularizer)

    def update(self):
        for layer in self.layers:
            layer.update(self.lr)



    def fit(self, epoch = 100, learning_rate = None):
        #update learning rate if it is given in fit
        if(learning_rate is not None):
            self.lr = learning_rate

        loss_train = []
        loss_dev = []
        #first get batch
        for i in range(epoch):

            self.batch.fit(self, size = self.batch_size)

            dev_loss = 0
            dev_accu = 0
            if(self.printInfo):

                if(self.dev_X is not None and self.dev_Y is not None):
                    dev_X_copy = np.copy(dev_X)
                    pred_dev = self.predict(self.dev_X)
                    dev_accu = np.mean( np.equal(np.argmax(self.dev_Y, 0), np.argmax(pred_dev, 0)))
                    dev_loss = self.cost.loss(dev_Y, pred_dev)
                    loss_dev.append(dev_loss)

                mean_loss_train = np.mean(self.batch.getLoss())
                loss_train.append(mean_loss_train)


                mean_accu_train = np.mean(self.batch.getAccuracy())
                print("epoch {}, train loss {}, train accur {}, val loss: {}, val accu: {}".format(i, mean_loss_train, mean_accu_train, dev_loss, dev_accu))

            self.plot.append(loss_train)

            self.plot.append(loss_dev)




    def predict(self, x):
        x = np.array(x)
        for layer in self.layers:
            x = layer.forward(x, training = False)#regularizer collect W during forward
        return x

    def test(self, test_x, test_y):
        pred_test = self.predict(test_x)
        test_accu = np.mean( np.equal(np.argmax(test_y, 0), np.argmax(pred_test, 0)))
        print("test accuracy: {}".format(test_accu))

    def plotLoss(self, x):

        plt.plot(np.arange(x), self.plot[0], label = "train_loss")
        plt.plot(np.arange(x), self.plot[1], label = "dev_loss")

        plt.xlabel("epoch")
        plt.ylabel("loss")

        plt.legend()
        plt.title("loss")

        plt.show()





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


def standardize_data(X):
    mean = np.mean(X, axis = 1, keepdims = True)
    var = np.var(X, axis = 1, keepdims = True)
    return (X-mean)/np.sqrt(var + 1e-8)

def onehot(label):
    return np.eye(np.max(label)+1)[label]

if __name__ == "__main__":
    X, label, toPredict= load_data() # X is(m, n_in)
    label = onehot(label)

    X = standardize_data(X.T)
    Y = label.T
    toPredict = standardize_data(toPredict.T)

    data = train_dev_test(X, Y, 0.8, 0.1, 0.1)
    (train_X, train_Y) = data[0]
    (dev_X, dev_Y) = data[1]
    (test_X, test_Y) = data[2]


    model = Model(X, Y, batch_size = 32, drop = 0.3, norm = standard(), optimizer = Adam())
    model.print_Info(True, 1)
    #model.set_dev(dev_X, dev_Y)
    #ini = He(), acti = relu()
    model.add_layer(192, ini = He(), acti = relu())
    model.add_layer(96, ini = He(), acti = relu())
    model.add_layer(48, ini = He(), acti = relu())
    model.add_last_layer(ini= He())

    loss = model.fit(epoch = 100, learning_rate = 0.0005)
    model.plotLoss(100)
    model.test(test_X, test_Y)
