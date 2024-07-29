import numpy as np
from layer import *
from optimizer import *
from activation import *
from loss import *

class SequentialNet:
    def __init__(self):
        self.loss_function = CrossEntropyLoss()
        self.conv1 = Convolutional(in_channels=1, out_channels=1, kernel_size=3, optimizer=SGD(learning_rate=0.01))
        self.fc1 = Linear(784, 64, activation=ReLU(), optimizer=SGD(learning_rate=0.01))
        self.fc2 = Linear(64, 32, activation=ReLU(), optimizer=SGD(learning_rate=0.01))
        self.fc3 = Linear(32, 10, activation=Softmax(), optimizer=SGD(learning_rate=0.01))

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    
    def __call__(self, x):
        return self.forward(x)
    
    def backward(self, x, y_true):
        y_pred = self.forward(x)
        loss = self.loss_function(y_true, y_pred)
        grad = self.loss_function.grad(y_true, y_pred)

        grad = self.fc3.backward(grad)
        grad = self.fc2.backward(grad)
        grad = self.fc1.backward(grad)
        return loss

    
    

