import numpy as np


class Sgd:
    def __init__(self, learning_rate):
        self.lr = learning_rate  #float

    def calculate_update(self, weight_tensor, gradient_tensor):
        updated_w_tensor = weight_tensor - self.lr * gradient_tensor
        return updated_w_tensor


class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.v is None:
            self.v = -1 * self.learning_rate * gradient_tensor
        else:
            self.v = np.subtract(self.momentum_rate * self.v, self.learning_rate * gradient_tensor)
        weight_tensor = np.add(weight_tensor, self.v)
        return weight_tensor


class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v = None
        self.r = None
        self.k = 1

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.v is None:
            self.v = (1 - self.mu) * gradient_tensor
        else:
            self.v = (self.mu * self.v) + ((1 - self.mu) * gradient_tensor)

        if self.r is None:
            self.r = (1 - self.rho) * np.multiply(gradient_tensor, gradient_tensor)
        else:
            self.r = (self.rho * self.r) + (1 - self.rho) * np.multiply(gradient_tensor, gradient_tensor)

        v_hat = self.v / (1 - (self.mu ** self.k))
        r_hat = self.r / (1 - (self.rho ** self.k))
        self.k += 1
        weight_tensor = weight_tensor - self.learning_rate * (v_hat / (np.sqrt(r_hat) + np.finfo(float).eps))
        return weight_tensor
