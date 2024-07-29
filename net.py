import numpy as np
from layer import *
from optimizer import *
from activation import *
from loss import *

class SequentialNet:
    def __init__(self):
        self.loss_function = None
        self.layers = None

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def __call__(self, x):
        return self.forward(x)
    
    def backward(self, x, y_true):
        y_pred = self.forward(x)
        loss = self.loss_function(y_true, y_pred)
        grad = self.loss_function.grad(y_true, y_pred)

        for layer in self.layers[::-1]: # list of layer reversed
            grad = layer.backward(grad)
        return loss

    
    

