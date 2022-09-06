import numpy as np
from Base import *

class Constant(BaseLayer):
    def __init__(self, ini_weight=0.1):
        super().__init__()
        self.ini_weight = ini_weight


    def initialize(self, weights_shape, fan_in=None, fan_out=None):
        weights_tensor = np.zeros(weights_shape)
        #weights_tensor = np.zeros((fan_in, fan_out))
        weights_tensor[:] = self.ini_weight
        return weights_tensor

class UniformRandom:
    def __init__(self):
        return

    def initialize(self, weights_shape, fan_in=None, fan_out=None):
        #weights_shape = (fan_in, fan_out)
        tensor = np.random.rand(*weights_shape)
        return tensor

class Xavier:
    def __init__(self):
        return

    def initialize(self, weights_shape, fan_in, fan_out):
         sigma = np.sqrt(2 / (fan_in + fan_out))
         rand_t = np.random.randn(*weights_shape)
         weight_ini = rand_t*sigma

         return weight_ini

class He:
    def __init__(self):
        return

    def initialize(self, weights_shape, fan_in, fan_out=None):
        #weights_tensor = np.zeros(weights_shape)
        rand_t = np.random.randn(*weights_shape)
        sigma = np.sqrt(2/fan_in)
        weight_ini = rand_t*sigma
        return weight_ini

if __name__ == "__main__":
    fanin = 400
    fanout=400

    x = Constant(0.1)
    a = x.initialize((400, 400),fanin,fanout)
    print(a)