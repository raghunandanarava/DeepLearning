import numpy as np


class Sgd:
    def __init__(self, learning_rate):
        self.lr = learning_rate  #float

    def calculate_update(self, weight_tensor, gradient_tensor):

        updated_w_tensor = weight_tensor - self.lr * gradient_tensor

        return updated_w_tensor


if __name__ == "__main__":

    print([1,2])
    sgd = Sgd(1.)
    result = sgd.calculate_update(1., 1.)
    print(result)
    #np.array([0.]))