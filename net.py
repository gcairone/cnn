import numpy as np
from layers import *
class SimpleNet:
    def __init__(self):
        self.conv1 = Convolutional(in_channels=1, out_channels=16, kernel_size=(3, 3))
        self.pool = Pooling(pool_size=(2, 2))
        self.conv2 = Convolutional(in_channels=16, out_channels=32, kernel_size=(3, 3))
        self.flatten = lambda x: x.reshape(x.shape[0], -1)  
        self.fc1 = Linear(32 * 6 * 6, 128)  
        self.fc2 = Linear(128, 10)  
    
    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.pool.forward(x)
        x = self.conv2.forward(x)
        x = self.pool.forward(x)
        x = self.flatten(x)
        x = self.fc1.forward(x)
        x = self.fc2.forward(x)
        return x
    
    def backward(self, x, y, learning_rate=0.001):
        def mse_loss(predictions, targets):
            return np.mean((predictions - targets) ** 2)
        logits =self.forward(x)
        
        batch_size = x.shape[0]
        grad_logits = 2.0 * (logits - y) / batch_size
        grad_fc2 = self.fc2.backward(self.fc1.forward(self.flatten(self.pool.forward(self.conv2.forward(self.pool.forward(self.conv1.forward(x)))))), grad_logits, learning_rate)
        grad_fc1 = self.fc1.backward(self.flatten(self.pool.forward(self.conv2.forward(self.pool.forward(self.conv1.forward(x))))), grad_fc2, learning_rate)
        grad_pool2 = self.pool.backward(self.conv2.backward(self.pool.forward(self.conv1.forward(x)), grad_fc1, learning_rate))
        grad_conv2 = self.conv2.backward(self.pool.forward(self.conv1.forward(x)), grad_pool2, learning_rate)
        grad_pool1 = self.pool.backward(self.conv1.backward(x, grad_conv2, learning_rate))
        grad_conv1 = self.conv1.backward(x, grad_pool1, learning_rate)
        
        loss = mse_loss(logits, y)
        return loss
    
    def train(self, x_train, y_train, epochs=10, learning_rate=0.001):
        for epoch in range(epochs):
            total_loss = 0
            for x, y in zip(x_train, y_train):
                loss = self.backward(x[np.newaxis, :], y[np.newaxis, :], learning_rate)
                total_loss += loss
            avg_loss = total_loss / len(x_train)
            print(f"Epoch [{epoch+1}/{epochs}], loss: {avg_loss:.4f}")
    
    def predict(self, x):
        logits = self.forward(x)
        return np.argmax(logits, axis=1)
