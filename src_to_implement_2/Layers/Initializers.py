import numpy as np


class Constant:
    def __init__(self, constant_value = 0.1):
        self.constant_value = constant_value

    def initialize(self, weights_shape, fan_in, fan_out):
        weights = np.full(weights_shape, self.constant_value)
        return weights


class UniformRandom:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        weights = np.random.uniform(0, 1, weights_shape)
        return weights


class Xavier:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        weights = np.random.normal(0, np.sqrt(2 / (fan_out + fan_in)), weights_shape)
        return weights


class He:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        weights = np.random.normal(0, np.sqrt(2 / fan_in), weights_shape)
        return weights
