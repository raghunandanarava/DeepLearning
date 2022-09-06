import numpy as np
from Base import BaseLayer


class Dropout(BaseLayer):
    def __init__(self, probability):
        super().__init__(False)
        self.probability = probability
        self.mask = None

    def forward(self, input_tensor):
        if self.testing_phase is False:
            self.mask = np.random.random(input_tensor.shape) < self.probability
            input_tensor = (input_tensor * self.mask) / self.probability
        return input_tensor

    def backward(self, error_tensor):
        return error_tensor * self.mask / self.probability
