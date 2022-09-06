import numpy as np


class SoftMax:
    def __init__(self):
        return

    def forward(self, input_tensor):
        # self.input_tensor = np.where(input_tensor > 0, input_tensor - np.max(input_tensor), input_tensor)

        input_tensor = np.asarray(input_tensor)

        # Taking the exponent of the matrix
        x_exp = np.exp(input_tensor - np.full(input_tensor.shape, np.max(input_tensor)))

        total_probabilities = np.sum(x_exp, axis=1, keepdims=True)

        # The corresponding probabilities matrix is obtained here np.sum(x_exp, axis=0)[:, None].T
        self.y_hat = np.divide(x_exp, total_probabilities)
        return self.y_hat

    def backward(self, error_tensor):
        error_tensor = np.multiply(self.y_hat, np.subtract(error_tensor, np.sum(np.multiply(self.y_hat, error_tensor), axis=1, keepdims=True)))
        return error_tensor

