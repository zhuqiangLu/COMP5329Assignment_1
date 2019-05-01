from Activation import tanh, sigmoid, softmax, relu
from BatchNormalizer import standard
from Layer import HiddenLayer
from Initializer import Xavier, He
from Cost import Cross_Entropy
from Dropout import Dropout
from Optimizer import Momentum, Nesterov, AdaGrad, AdaDelta, RMSProp, Adam
from Regularizer import L1, L2
from SGD import Batch
import time
import h5py


import numpy as np
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

        #create batches
        self.batch = Batch(training_data, training_label)
        self.batch_size = batch_size

        #place holder for general info
        self.m = training_data.shape[1]
        self.cv_X = None
        self.cv_Y = None
        self.classes = training_label.shape[0]
        self.dims = [training_data.shape[0]]
        self.lr = learning_rate
        self.epoch = 100

        #place holder for layers
        self.layers = []

        #optimizer
        self.optimizer = optimizer

        #regularizer
        self.drop = drop
        self.norm = norm
        self.regularizer = regularizer

        #set optimizer to the Batch Normalizer
        if(self.optimizer is not None and self.norm is not None):
            self.norm.set_optimizer(self.optimizer.clone())

        #cost function
        self.cost = cost

        #plot infos
        self.printInfo = False
        self.printAt = 1
        self.Loss_plot = []
        self.Accu_plot = []



    def add_layer(self, n_out, ini = Xavier(), acti = relu(), drop = None):

        #override the universal drop
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
        #The activation function is softmax for the last layer
        layer.setActivation(softmax())

        #last layer dose not need Dropout
        layer.setDropout(drop = 0)
        if(self.optimizer != None):
            layer.setOptimizer(self.optimizer.clone())

        self.layers.append(layer)

    def cross_validate(self, cv_X, cv_Y):
        #set cross validation data
        self.cv_X = cv_X
        self.cv_Y = cv_Y

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
        print("<==============Start training================>")
        self.epoch = epoch

        #update learning rate if it is given in fit
        if(learning_rate is not None):
            self.lr = learning_rate

        #place holder for info to plot
        total_loss_train = []
        total_loss_cv = []
        total_accu_train = []
        total_accu_cv = []

        start = time.time()

        for i in range(epoch):
            #go through all mini_batches
            self.batch.fit(self, size = self.batch_size)

            #get mean loss of all batch losses
            mean_loss_train = np.mean(self.batch.getLoss())
            mean_accu_train = np.mean(self.batch.getAccuracy())
            total_loss_train.append(mean_loss_train)
            total_accu_train.append(mean_accu_train)


            #if we have cross validation set, we log the plot info
            cv_loss = 0
            cv_accu = 0
            if(self.cv_X is not None and self.cv_Y is not None):

                pred_cv = self.predict(self.cv_X)
                cv_loss = self.cost.loss(self.cv_Y, pred_cv)
                cv_accu = np.mean( np.equal(np.argmax(self.cv_Y, 0), np.argmax(pred_cv, 0)))

                total_loss_cv.append(cv_loss)
                total_accu_cv.append(cv_accu)




            self.Loss_plot.append(total_loss_train)
            self.Loss_plot.append(total_loss_cv)
            self.Accu_plot.append(total_accu_train)
            self.Accu_plot.append(total_accu_cv)

            if(self.printInfo and i%self.printAt == 0):
                print("epoch {}, train loss {:.5f}, train accur {:.3%}, val loss: {:.5f}, val accu: {:.3%}".format(i+1, mean_loss_train, mean_accu_train, cv_loss, cv_accu))

        end = time.time()
        if(self.printInfo):
            s = end - start
            print("Total training time {:.3f} s".format(s))

    def predict(self, x):
        x = np.array(x)
        for layer in self.layers:
            x = layer.forward(x, training = False)#regularizer collect W during forward
        return x


    def test(self, test_x, test_y):
        if(test_x is None):
            return
        pred_test = self.predict(test_x)
        test_accu = np.mean( np.equal(np.argmax(test_y, 0), np.argmax(pred_test, 0)))
        print("Test accuracy: {:.2%}".format(test_accu))



    def plot(self, accu = True, loss = True):
        if(accu or loss):

            #get x axis
            x = np.arange(self.epoch+1)
            #exclude epoch 0
            x = x[1:]

            #get subplot number
            subplot_number = 1
            i = 1
            if(accu and loss):
                subplot_number += 1

            #create figure
            plt.figure(1)

            if(accu):
                plt.subplot(subplot_number, 1, i)
                plt.plot(x, self.Accu_plot[0], label = "train_accu")
                if(self.cv_X is not  None):
                    plt.plot(x, self.Accu_plot[1], label = "cv_accu")

                plt.xlabel("epoch")
                plt.ylabel("accu")

                plt.legend()

                i+=1



            if(loss):
                plt.subplot(subplot_number, 1, i)
                plt.plot(x, self.Loss_plot[0], label = "train_loss")
                if(self.cv_X is not  None):
                    plt.plot(x, self.Loss_plot[1], label = "cv_loss")

                plt.xlabel("epoch")
                plt.ylabel("loss")

                plt.legend()

            plt.show()
