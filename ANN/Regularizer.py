import numpy as np

class L2(object):
    def __init__(self, lamda = 1):
        self.lamda = lamda
        self.loss= 0

    def reset(self):
        self.loss = 0

    def collectW(self, W):
        self.loss += np.sum(np.square(W))

    def update_grad_W(self, grad_W, W, m):
        return grad_W + (self.lamda/(2*m))*W

class L1(object):
    def __init__(self, lamda = 1):
        self.lamda = lamda
        self.loss = 0

    def reset(self):
        self.loss = 0

    def collectW(self, W):
        self.loss += np.sum(np.abs(W))

    def update_grad_W(self, grad_W, W, m):
        return grad_W + (self.lamda/(m))*(np.sign(W))


class dumbRegularizer(object):
    def __init__(self, lamda = 1):
        self.lamda = lamda
        self.loss = 0

    def reset(self):
        pass

    def collectW(self, W):
        pass


    def update_grad_W(self, grad_W, W, m):
        return grad_W
