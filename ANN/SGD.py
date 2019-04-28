import numpy as np


class Batch(object):
    def __init__(self, X, Y):
        #assume X is of the shape (n_features, n_example)
        #Y is of the shape (n_classes, n_example)
        self.x_features = X.shape[0]
        self.y_classes = Y.shape[0]
        self.m = X.shape[1]
        self.map = np.vstack((X,Y))
        self.loss = []
        self.accuracy = []

    def shuffle(self, boo = True):

        #shuffle only shuffles the array along the first axis
        #must reshape map first
        m_t = np.copy(self.map.T)
        np.random.shuffle(m_t)
        self.map = np.copy(m_t.T)



    def getX(self):
        return self.map[:self.x_features, :]

    def getY(self):
        return self.map[self.x_features:, :]

    def getLoss(self):
        return self.loss

    def getAccuracy(self):
        return self.accuracy



    def reset(self):
        self.loss = []
        self.accuracy = []



    def fit(self, model, size = None):

        self.reset()

        if(size == None):
            size = self.m


        map_c = np.copy(self.map)
        self.shuffle()
        batch_num = self.m//size


        for i in range(batch_num):

            start = i * size
            end = start + size
            shuff_X = self.getX()
            shuff_Y = self.getY()
            mini_X = shuff_X[:, start:end]
            mini_Y = shuff_Y[:, start:end]


            mini_Y_hat = model.forward(mini_X, training = True)

            self.loss.append( model.cost.loss(mini_Y, mini_Y_hat) + model.get_reg_loss())

            #compute accuracy for this mini batch
            self.accuracy.append(np.mean( np.equal(np.argmax(mini_Y, 0), np.argmax(mini_Y_hat, 0))))

            mini_dz = model.cost.dz(mini_Y, mini_Y_hat)
            model.backward(mini_dz)
            model.update()
