import numpy as np

class L2(object):
    def __init__(self, lamda = 1):
        self.lamda = lamda
        self.loss= 0

    def reset(self):
        self.loss = 0

    def forward(self, W):
        self.loss += np.sum(np.square(W))

    def get_loss(self, m):
        return self.loss * (self.lamda/(2*m))

    def backward(self, grad_W, W, m):
        return grad_W + (self.lamda/(m))* (W)

class L1(object):
    def __init__(self, lamda = 1):
        self.lamda = lamda
        self.loss = 0

    def reset(self):
        self.loss = 0

    def forward(self, W):
        self.loss += np.sum(np.abs(W))

    def get_loss(self, m):
        return self.loss * self.lamda/m

    def backward(self, grad_W, W, m):
        return grad_W + (self.lamda/(m))*(np.sign(W))
