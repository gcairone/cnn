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
        out_conv1 = self.conv1.forward(x)
        out_pool1 = self.pool.forward(out_conv1)
        out_conv2 = self.conv2.forward(out_pool1)
        out_pool2 = self.pool.forward(out_conv2)
        out_flat = self.flatten(out_pool2)
        out_fc1 = self.fc1.forward(out_flat)
        out_fc2 = self.fc2.forward(out_fc1)

        grad_fc2 = self.fc2.backward(out_fc1, grad_logits, learning_rate)
        grad_fc1 = self.fc1.backward(out_flat, grad_fc2, learning_rate)
        grad_flat = self.flatten.backward(out_pool2, grad_fc1)
        grad_pool2 = self.pool.backward(out_conv2, grad_flat)
        grad_conv2 = self.conv2.backward(out_pool1, grad_pool2, learning_rate)
        grad_pool1 = self.pool.backward(out_conv1, grad_conv2)
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
