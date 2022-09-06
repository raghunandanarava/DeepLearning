import numpy as np

class Optimizer:
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer

class Sgd(Optimizer):
    def __init__(self, learning_rate):
        super().__init__()
        self.lr = learning_rate  #float

    def calculate_update(self, weight_tensor, gradient_tensor):
        weight_tensor = np.asarray(weight_tensor)
        gradient_tensor = np.asarray(gradient_tensor)

        if self.regularizer is not None:
            updated_w_tensor = weight_tensor - self.lr * self.regularizer.calculate_gradient(weight_tensor) \
                               - self.lr * gradient_tensor
        else:
            updated_w_tensor = weight_tensor - self.lr * gradient_tensor
        return updated_w_tensor

class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.lr = learning_rate
        self.mr = momentum_rate
        self.velocity = None

    def calculate_update(self, weight_tensor, gradient_tensor):

        weight_tensor = np.asarray(weight_tensor)
        gradient_tensor = np.asarray(gradient_tensor)

        if self.velocity is None:
            self.velocity = -self.lr * gradient_tensor
        else:
            self.velocity = self.mr * self.velocity - self.lr * gradient_tensor

        if self.regularizer is not None:
            updated_w_t = weight_tensor + self.velocity - self.lr * self.regularizer.calculate_gradient(weight_tensor)
        else:
            updated_w_t = weight_tensor + self.velocity
        return updated_w_t


class Adam(Optimizer):
    def __init__(self, learning_rate, beta1, beta2):
        super().__init__()
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.velocity = None
        self.r = None
        self.k = 1
        self.epsilon = np.finfo(float).eps

    def calculate_update(self, weight_tensor, gradient_tensor):

        weight_tensor = np.asarray(weight_tensor)
        gradient_tensor = np.asarray(gradient_tensor)

        #contains bias correction
        if self.velocity is None:
            self.velocity = (1 - self.beta1) * gradient_tensor
        else:
            self.velocity = (self.beta1 * self.velocity + (1 - self.beta1) * gradient_tensor)

        if self.r is None:
            self.r = (1-self.beta2) * np.multiply(gradient_tensor, gradient_tensor)
        else:
            self.r = (self.beta2 * self.r + (1-self.beta2) * np.multiply(gradient_tensor, gradient_tensor))

        velocity_hat = self.velocity / (1 - np.power(self.beta1, self.k))
        r_hat = self.r / (1 - np.power(self.beta2, self.k))

        constraint = self.lr * velocity_hat/(np.sqrt(r_hat) + self.epsilon)
        if self.regularizer is not None:
            new_weight = weight_tensor - constraint - self.lr * self.regularizer.calculate_gradient(weight_tensor)
        else:
            new_weight = weight_tensor - constraint
        self.k += 1
        return new_weight



