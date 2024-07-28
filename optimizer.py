import numpy as np

class Optimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def step(self, params, grads):
        raise NotImplementedError("This method should be overridden by subclasses")

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)

    def step(self, params, grads):
        for param, grad in zip(params, grads):
            param -= self.learning_rate * grad

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        lr_t = self.learning_rate * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)

        for param, grad in zip(params, grads):
            if param not in self.m:
                self.m[param] = np.zeros_like(grad)
                self.v[param] = np.zeros_like(grad)

            self.m[param] = self.beta1 * self.m[param] + (1 - self.beta1) * grad
            self.v[param] = self.beta2 * self.v[param] + (1 - self.beta2) * (grad ** 2)

            m_hat = self.m[param] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param] / (1 - self.beta2 ** self.t)

            param -= lr_t * m_hat / (np.sqrt(v_hat) + self.epsilon)
