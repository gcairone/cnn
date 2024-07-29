import numpy as np
from activation import ActivationFunction
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


class Convolutional(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation=None, optimizer=None):
        super().__init__(activation, optimizer)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
        self.bias = np.zeros((out_channels, 1, 1))
        
        self.params = [self.weights, self.bias]

    def _pad(self, x):
        if self.padding > 0:
            return np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        return x

    def forward(self, x):
        self.input = x
        batch_size, _, in_height, in_width = x.shape
        
        x_padded = self._pad(x)
        
        out_height = (in_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (in_width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        output = np.zeros((batch_size, self.out_channels, out_height, out_width))
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size
                
                receptive_field = x_padded[:, :, h_start:h_end, w_start:w_end]
                output[:, :, i, j] = np.sum(receptive_field[:, np.newaxis, :, :, :] * self.weights, axis=(2, 3, 4))
        
        output += self.bias
        
        if self.activation:
            return self.activation(output)
        return output

    def backward(self, grad_output):
        batch_size, _, out_height, out_width = grad_output.shape
        
        if self.activation:
            grad_output = grad_output * self.activation.der(self.forward(self.input))
        
        grad_weights = np.zeros_like(self.weights)
        grad_bias = np.sum(grad_output, axis=(0, 2, 3)).reshape(self.out_channels, 1, 1)
        grad_input = np.zeros_like(self.input)
        
        padded_input = self._pad(self.input)
        padded_grad_input = self._pad(grad_input)
        
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size
                
                receptive_field = padded_input[:, :, h_start:h_end, w_start:w_end]
                for k in range(self.out_channels):
                    grad_weights[k] += np.sum(receptive_field * grad_output[:, k:k+1, i:i+1, j:j+1], axis=0)
                
                for c in range(self.in_channels):
                    padded_grad_input[:, c, h_start:h_end, w_start:w_end] += np.sum(
                        self.weights[:, c] * grad_output[:, :, i:i+1, j:j+1], axis=1
                    )
        
        if self.padding > 0:
            grad_input = padded_grad_input[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            grad_input = padded_grad_input
        
        self.grads = [grad_weights, grad_bias]
        self.update_params()
        
        return grad_input
