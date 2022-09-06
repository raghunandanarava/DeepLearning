import numpy as np


class Sigmoid:
    def __init__(self):
        self.activation = None

    def forward(self, input_tensor):
        self.activation = 1 / (1 + np.exp(-1 * input_tensor))
        return self.activation

    def backward(self, error_tensor):
        return (self.activation * (1 - self.activation)) * error_tensor
