import numpy as np

class ActivationFunction:
    def __call__(self, x):
        raise NotImplementedError("This method should be overridden by subclasses")

    def der(self, x):
        raise NotImplementedError("This method should be overridden by subclasses")

class ReLU(ActivationFunction):
    def __call__(self, x):
        return np.maximum(0, x)

    def der(self, x):
        return np.where(x > 0, 1, 0)

class Sigmoid(ActivationFunction):
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def der(self, x):
        sig = self.__call__(x)
        return sig * (1 - sig)

class Softmax(ActivationFunction):
    def __call__(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def der(self, x):
        # softmax gradient not normally used
        softmax = self.__call__(x)
        return softmax * (1 - softmax)
