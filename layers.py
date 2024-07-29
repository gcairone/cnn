import numpy as np
from functions import ActivationFunction
from optimizer import *


class Layer:
    def __init__(self, activation, optimizer):
        self.input = None
        self.activation = activation
        self.optimizer = optimizer
        self.params = []
        self.grads = []


    def forward(self, x):
        raise NotImplementedError("This method should be overridden by subclasses")

    def __call__(self, x):
        return self.forward(x)

    def backward(self, x, grad_output, learning_rate):
        raise NotImplementedError("This method should be overridden by subclasses")

    def update_params(self):
        if self.optimizer:
            self.optimizer.step(self.params, self.grads)

class Linear(Layer):
    def __init__(self, in_features, out_features, activation=None, optimizer=None):
        super().__init__(activation, optimizer)
        self.in_features = in_features
        self.out_features = out_features
        self.weights = np.random.randn(in_features, out_features) * 0.01
        self.bias = np.random.randn(out_features) * 0.01

        self.params = [self.weights, self.bias]

    def forward(self, x):
        self.input = x
        z = x @ self.weights + self.bias  # Output lineare
        if self.activation:
            return self.activation(z)
        return z
    
    def backward(self, grad_output):
        if self.activation:
            grad_output = grad_output * self.activation.der(self.input @ self.weights + self.bias)

        grad_weights = self.input.T @ grad_output
        grad_bias = np.sum(grad_output, axis=0)
        grad_input = grad_output @ self.weights.T

        self.grads = [grad_weights, grad_bias]
        self.update_params()
        
        return grad_input




"""
class Convolutional:
    def __init__(self, in_channels, out_channels, kernel_size):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weights = np.zeros(shape=(kernel_size, kernel_size))  
        # self.bias = ...     

    def forward(self, x):
        batch_size, _, input_height, input_width = x.shape
        _, _, kernel_height, kernel_width = self.weights.shape
        
        output_height = input_height - kernel_height + 1
        output_width = input_width - kernel_width + 1
        
        output = np.zeros((batch_size, self.out_channels, output_height, output_width))
        
        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for c_in in range(self.in_channels):
                    for i in range(output_height):
                        for j in range(output_width):
                            output[b, c_out, i, j] += np.sum(
                                x[b, c_in, i:i+kernel_height, j:j+kernel_width] * self.weights[c_out, c_in]
                            )
        
        # output += self.bias.reshape(1, -1, 1, 1)  
        
        return output
    
    def backward(self, x, grad_output, learning_rate):
        batch_size, _, input_height, input_width = x.shape
        _, _, kernel_height, kernel_width = self.weights.shape
        _, _, output_height, output_width = grad_output.shape
        
        grad_input = np.zeros_like(x)
        grad_weights = np.zeros_like(self.weights)
        
        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for c_in in range(self.in_channels):
                    for i in range(output_height):
                        for j in range(output_width):
                            grad_weights[c_out, c_in] += np.sum(
                                x[b, c_in, i:i+kernel_height, j:j+kernel_width] * grad_output[b, c_out, i, j]
                            )
                            grad_input[b, c_in, i:i+kernel_height, j:j+kernel_width] += (
                                self.weights[c_out, c_in] * grad_output[b, c_out, i, j]
                            )
        
        self.weights -= learning_rate * grad_weights
        
        return grad_input

class Pooling:
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.cache = None
    
    def forward(self, x):
        self.cache = x  
        
        batch_size, num_channels, input_height, input_width = x.shape
        pool_height, pool_width = self.pool_size
        
        output_height = input_height // pool_height
        output_width = input_width // pool_width
        
        output = np.zeros((batch_size, num_channels, output_height, output_width))
        
        for b in range(batch_size):
            for c in range(num_channels):
                for i in range(output_height):
                    for j in range(output_width):
                        window = x[b, c, i*pool_height:(i+1)*pool_height, j*pool_width:(j+1)*pool_width]
                        output[b, c, i, j] = np.max(window)
        
        return output
    
    def backward(self, grad_output):
        x = self.cache
        batch_size, num_channels, input_height, input_width = x.shape
        pool_height, pool_width = self.pool_size
        
        output_height, output_width = grad_output.shape[2], grad_output.shape[3]
        
        grad_input = np.zeros_like(x)
        
        for b in range(batch_size):
            for c in range(num_channels):
                for i in range(output_height):
                    for j in range(output_width):
                        window = x[b, c, i*pool_height:(i+1)*pool_height, j*pool_width:(j+1)*pool_width]
                        max_value = np.max(window)
                        mask = (window == max_value)
                        grad_input[b, c, i*pool_height:(i+1)*pool_height, j*pool_width:(j+1)*pool_width] += mask * grad_output[b, c, i, j]
        
        return grad_input

"""



