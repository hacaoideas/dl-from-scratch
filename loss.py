import numpy as np 
import activations as ac 

#https://en.wikipedia.org/wiki/Loss_functions_for_classification

class Loss:
    def loss(self, y_true, y_pred):
        raise NotImplementedError()

    def gradient(self, y, y_pred):
        raise NotImplementedError()

    def accuracy(self, y, y_pred):
        return 0


class SquareLoss(Loss):
    def __init__(self):
        pass

    def loss(self, y, y_pred):
        return 0.5*np.power((y-y_pred),2)

    def gradient(self, y, y_pred):
        return -(y-y_pred)

class CrossEntropy(Loss):
    def __init__(self):
        pass

    def loss(self, y, p):

        p = np.clip(p, 1e-15, 1 - 1e-15)
        #values outside the interval are clipped to the
        #interval's edge
        return - y *np.log(p) - (1-y) * np.log(1-p)

    def gradient(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y/p) + (1 - y)/(1-p)

    def accuracy(self, y, p):
        pass 