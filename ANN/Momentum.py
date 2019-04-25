import numpy as np


class Standard_Momentum(object):
    def __init__(self, momentum_term = 0.9):
        self.gama = 0.9
        self.v_W = 0
        self.v_b = 0

    def clone(self):
        return Standard_Momentum(self.gama)

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
        return Standard_Momentum(self.gama)

    def update_W(self, lr, W, grad_W):
        self.v_W = (self.gama * self.v_W) + (lr * grad_W) * (W - (self.gama * self.v_W))
        return W - self.v_W

    def update_b(self, lr, b, grad_b):
        self.v_b = (self.gama * self.v_b) + (lr * grad_b) * (b - (self.gama * self.v_b))
        return b - self.v_b
