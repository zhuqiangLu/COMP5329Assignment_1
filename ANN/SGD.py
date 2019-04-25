import numpy as np


class Batch(object):
    def __init__(self, X, Y):
        #assume X is of the shape (n_features, n_example)
        #Y is of the shape (n_classes, n_example)
        self.x_features = X.shape[0]
        self.y_classes = Y.shape[0]
        self.m = X.shape[1]
        self.map = np.vstack((X,Y))
        self.loss = 0
        self.accuracy = 0

    def shuffle(self):
        #shuffle only shuffles the array along the first axis
        #must reshape map first
        map_t = self.map.T
        np.random.shuffle(map_t)
        self.map = map_t.T

    def getX(self):
        return self.map[:self.x_features, :]

    def getY(self):
        return self.map[self.x_features:, :]

    def getLoss(self):
        return self.loss

    def getAccuracy(self):
        return self.accuracy

    def reset(self):
        self.loss = 0
        self.accuracy = 0



    def fit(self, model, size = None):
        #shuffle the batch before forward
        self.reset()

        if(size == None):
            size = self.m
        self.shuffle()
        batch_num = self.m//size

        for i in range(batch_num):
            start = i * size
            end = (i+1) * size

            mini_X = self.getX()[:, start:end]
            mini_Y = self.getY()[:, start:end]

            if(i == batch_num):
                mini_X = self.getX()[:, start:]
                mini_Y = self.getY()[:, start:]


            mini_Y_hat = model.forward(mini_X)
            self.loss += model.cost.loss(mini_Y, mini_Y_hat) + model.regularizer.loss
            #compute accuracy for this mini batch
            self.accuracy += np.mean( np.equal(np.argmax(mini_Y, 0), np.argmax(mini_Y_hat, 0)))
            mini_dz = model.cost.dz(mini_Y, mini_Y_hat)
            model.backward(mini_dz)

            model.update()

        self.loss = self.loss/batch_num
        self.accuracy = self.accuracy/batch_num
