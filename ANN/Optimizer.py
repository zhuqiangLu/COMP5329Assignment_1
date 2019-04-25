import numpy as np


class Momentum(object):
    def __init__(self, momentum_term = 0.9):
        self.gama = 0.9
        self.v_W = 0
        self.v_b = 0

    def clone(self):
        return Momentum(self.gama)

    def update_W(self, lr, W, grad_W):
        self.v_W = (self.gama * self.v_W) + (lr * grad_W)
        return W - self.v_W

    def update_b(self, lr, b, grad_b):
        self.v_b =(self.gama * self.v_b) + (lr * grad_b)
        return b - self.v_b


class Nesterov(object):
    def __init__(self, momentum_term = 0.9):
        self.gama = 0.9
        self.v_W = 0
        self.v_b = 0

    def clone(self):
        return Momentum(self.gama)

    def update_W(self, lr, W, grad_W):
        self.v_W = (self.gama * self.v_W) + (lr * grad_W) * (W - (self.gama * self.v_W))
        return W - self.v_W

    def update_b(self, lr, b, grad_b):
        self.v_b = (self.gama * self.v_b) + (lr * grad_b) * (b - (self.gama * self.v_b))
        return b - self.v_b

class AdaGrad(object):

    def __init__(self):
        self.epsilon = 1e-6
        self.G_W = 0
        self.G_b = 0


    def update_W(self, lr, W, grad_W):

        self.G_W += np.square(grad_W)
        return W - (lr/(np.sqrt(self.G_W) + self.epsilon)) * grad_W

    def update_b(self, lr, b, grad_b):
        self.G_b += np.square(grad_b)
        return b - (lr/(np.sqrt(self.G_b) + self.epsilon)) * grad_b

class RMSProp(object):
    def __init__(self, decay = 0.9):
        self.epsilon = 1e-6
        self.G_W = 0
        self.G_b = 0
        self.decay = decay


    def update_W(self, lr, W, grad_W):

        self.G_W += (self.decay * (self.G_W)) + ((1 - self.decay) * np.square(grad_W))
        return W - (lr / np.sqrt(self.G_W + self.epsilon)) * grad_W

    def update_b(self, lr, b, grad_b):

        self.G_b += (self.decay * (self.G_b)) + ((1 - self.decay) * np.square(grad_b))
        return b - (lr / np.sqrt(self.G_b + self.epsilon)) * grad_b
