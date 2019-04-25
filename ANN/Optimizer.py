import numpy as np

class Momentum(object):
    def __init__(self, method = "standard",  momentum_term = 0.9):
        self.gama = 0.9
        self.method = method.lower()
        self.v_W = 0
        self.v_b = 0

    def clone(self):
        return Momentum(method = self.method, momentum_term = self.gama)

    def update_W(self, lr, W, grad_W):
        if(self.method == "standard"):
            self.v_W = (self.gama * self.v_W) + (lr * grad_W)
            return W - self.v_W
        elif(self.method == "nesterov"):
            self.v_W = (self.gama * self.v_W) + (lr * grad_W) * (W - (self.gama * self.v_W))
            return W - self.v_W
        else:
            raise ValueError('unknown mementum method ',self.mothod)


    def update_b(self, lr, b, grad_b):
        if(self.method == "standard"):
            self.v_b =(self.gama * self.v_b) + (lr * grad_b)
            return b - self.v_b
        elif(self.method == "nesterov"):
            self.v_b = (self.gama * self.v_b) + (lr * grad_b) * (b - (self.gama * self.v_b))
            return b - self.v_b
        else:
            raise ValueError('unknown mementum method ',self.mothod)




#ALRM stands for adaptive learing rate methods
class ALRM(object):

    def __init__(self, method = "adam", decay = 0.9):
        self.method = method.lower()
        self.epsilon = 1e-6
        self.G_W = 0
        self.G_b = 0
        self.decay = decay

    def update_W(self, lr, W, grad_W):
        if(self.method == "adagrad"):
            self.G_W += np.square(grad_W)
            return W - ((lr/np.sqrt(self.G_W + self.epsilon))*grad_W)
        elif(self.method == "rmsprop"):
            self.G_W += (self.decay * (self.G_W)) + ((1 - self.decay) * np.square(grad_W))

    def update_b(self, lr, b, grad_b):
        if(self.method == "adagrad"):
            self.G_b += np.square(grad_b)
            return b - ((lr/np.sqrt(self.G_b + self.epsilon))*grad_b)
