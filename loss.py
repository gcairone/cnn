import numpy as np

class LossFunction:
    def __call__(self, y_true, y_pred):
        raise NotImplementedError("This method should be overridden by subclasses")

    def grad(self, y_true, y_pred):
        raise NotImplementedError("This method should be overridden by subclasses")

class MSELoss(LossFunction):
    def __call__(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def grad(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size

class CrossEntropyLoss(LossFunction):
    def __call__(self, y_true, y_pred):
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def grad(self, y_true, y_pred):
        # Clip predictions to avoid division by zero
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        return (y_pred - y_true) / (y_pred * (1 - y_pred))
