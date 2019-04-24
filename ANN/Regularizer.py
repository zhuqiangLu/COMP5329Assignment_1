import numpy as np

class L2(object):
    def __init__(self):
        self.lamda = 1
        self.loss= 0

    def setLamda(self, lamda):
        self.lamda = lamda

    def reset(self):
        self.loss = 0

    def collectW(self, W):
        self.loss += np.sum(np.square(W))

    def update(self, grad_W, W, m):
        return grad_W + (self.lamda/(2*m))*W

class L1(object):
    def __init__(self):
        self.lamda = 1
        self.loss = 0

    def setLamda(self, lamda):
        self.lamda = lamda

    def reset(self):
        self.loss = 0

    def collectW(self, W):
        self.loss += np.sum(np.abs(W))

    def update(self, grad_W, W, m):
        return grad_W + (self.lamda/(m))*(self.W/np.abs(W))


class dumbRegularizer(object):
    def __init__(self):
        self.lamda = 1
        self.loss = 0

    def setLamda(self, lamda):
        pass

    def reset(self):
        pass

    def collectW(self, W):
        pass


    def update(self, grad_W, W, m):
        return grad_W
