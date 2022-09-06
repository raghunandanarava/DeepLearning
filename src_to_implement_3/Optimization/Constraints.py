import numpy as np
from numpy import linalg as LA

class L2_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        grd = weights * self.alpha
        return grd

    def norm(self, weights):
        fro = np.power(LA.norm(weights), 2) * self.alpha
        return fro


class L1_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        grd = self.alpha * np.sign(weights)
        return grd

    def norm(self, weights):
        fro = np.power(LA.norm(weights), 2) * self.alpha
        return fro

if __name__ == "__main__":
    weights_tensor = np.ones((4,5))
    weights_tensor[1:3, 2:4] *= -1
    print(weights_tensor)