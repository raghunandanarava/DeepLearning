import numpy as np
from Layers import SoftMax


class CrossEntropyLoss:
    def __init__(self):
        self.input_tensor = None
        return

    def forward(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        loss = -1 * np.sum(np.where(label_tensor == 1, np.log(self.input_tensor + np.finfo(float).eps), 0))
        return loss

    def backward(self, label_tensor):
        error_tensor = -1 * np.divide(label_tensor, self.input_tensor)
        return error_tensor
