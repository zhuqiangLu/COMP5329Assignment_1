from Activation import tanh, sigmoid, softmax, relu
from BatchNormalizer import standard
from Layer import HiddenLayer
from Initializer import Xavier, He
from Cost import Cross_Entropy
from Dropout import Dropout
from Optimizer import Momentum, Nesterov, AdaGrad, AdaDelta, RMSProp, Adam
from Regularizer import L1, L2
from SGD import Batch
from Model import Model
import numpy as np
import h5py

def load_data():
    with h5py.File('Assignment-1-Dataset/train_128.h5','r') as H:
        data = np.copy(H['data'])
    with h5py.File('Assignment-1-Dataset/train_label.h5','r') as H:
        label = np.copy(H['label'])
    with h5py.File('Assignment-1-Dataset/test_128.h5','r') as H:
        test = np.copy(H['data'])
    return data,label,test

def standardize_data(X):
    mean = np.mean(X, axis = 1, keepdims = True)
    var = np.var(X, axis = 1, keepdims = True)
    return (X-mean)/np.sqrt(var + 1e-8)


def split_data(X, Y, train_rate, cv_rate, test_rate):
    m = X.shape[1]
    train = round(m*train_rate)
    cv = train + round(m*cv_rate)
    test = cv + round(m*test_rate)
    return [(X[:,:train], Y[:,:train]),(X[:,train:cv], Y[:, train:cv]),(X[:, cv:], Y[:, cv:])]

def onehot(label):
    return np.eye(np.max(label)+1)[label]


def main():
    X, label, toPredict= load_data() # X is(m, n_in)
    label = onehot(label)

    X = standardize_data(X.T)
    Y = label.T
    toPredict = standardize_data(toPredict.T)

    data = split_data(X, Y, 0.8, 0.1, 0.1)
    (train_X, train_Y) = data[0]
    (cv_X, cv_Y) = data[1]
    (test_X, test_Y) = data[2]

    model = Model(train_X, train_Y, batch_size = 32, drop = 0.3, norm = standard(), optimizer = Adam())
    model.print_Info(True, 1)
    model.cross_validate(cv_X, cv_Y)
    model.add_layer(192, ini = He(), acti = relu())
    model.add_layer(96, ini = He(), acti = relu())
    model.add_layer(48, ini = He(), acti = relu())
    model.add_last_layer(ini= He())
    model.fit(epoch = 50, learning_rate = 0.0005)
    model.test(test_X, test_Y)
    model.plot()


main()
