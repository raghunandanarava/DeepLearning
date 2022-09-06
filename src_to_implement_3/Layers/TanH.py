import numpy as np


class TanH:
    def __init__(self):
        self.activation = None

    def forward(self, input_tensor):
        self.activation = np.tanh(input_tensor)
        return self.activation

    def backward(self, error_tensor):
        return (1 - np.power(self.activation, 2)) * error_tensor
