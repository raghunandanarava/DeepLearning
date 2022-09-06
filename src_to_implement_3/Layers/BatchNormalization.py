import numpy as np
from Base import *
import Helpers

class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)
        self.mean_moving = None
        self.var_moving = None
        self.gradient_weights = None
        self.gradient_bias = None
        self._optimizer = None

    def forward(self, input_tensor):
        epsilon = np.finfo(float).eps
        self.input_tensor = input_tensor

        if len(input_tensor.shape) > 2:
            input_tensor = self.reformat(input_tensor)

        mean = np.mean(input_tensor, axis=0)
        #print('the mean is', mean)
        var = np.var(input_tensor, axis=0)
        #print('the variance', var)


        if not self.testing_phase:
            #training
            output = np.subtract(input_tensor, mean) / (np.sqrt(var + epsilon))
            self.output = output
            #print('middle', np.var(np.subtract(input_tensor, mean), axis=0))
            #print('middle', np.var(output, axis=0))
            output_t = np.multiply(output, self.weights) + self.bias
            #print('middle', np.var(output_t, axis=0))

            decay = 0.8
            if self.mean_moving is None:
                self.mean_moving = mean
            else:
                self.mean_moving = decay * self.mean_moving + (1 - decay) * mean

            if self.var_moving is None:
                self.var_moving = var
            else:
                self.var_moving = decay * self.var_moving + (1 - decay) * var
        else:
            output = np.subtract(input_tensor, self.mean_moving) / (np.sqrt(self.var_moving + epsilon))
            self.output = output
            #print(self.bias.shape)
            output_t = np.multiply(self.weights, output) + self.bias
            #print('middle', output_t)

        #print('result', output_t.shape)
        #print('input', len(input_tensor.shape))
        if len(self.input_tensor.shape) > 2:
            output_t = self.reformat(output_t)

        return output_t

    def get_op(self):
        return self._optimizer

    def set_op(self, op):
        self._optimizer = op

    optimizer = property(get_op, set_op)

    def backward(self, error_tensor):
        #axis is batch
        #print(self.input_tensor.shape)
        #print(error_tensor)
        input_t = self.input_tensor
        if len(self.input_tensor.shape) > 2:
            error_tensor = self.reformat(error_tensor)
            input_t = self.reformat(self.input_tensor)

        mean = np.mean(input_t, axis=0)
        #print('the mean is', mean)
        var = np.var(input_t, axis=0)
        #print('the variance', var)
        error_tensor_prev = Helpers.compute_bn_gradients(error_tensor, input_t, self.weights, mean, var)
        print(error_tensor_prev.shape)

        self.gradient_bias = np.sum(error_tensor, axis=0)
        #P = np.multiply(error_tensor, self.output_t)
        self.gradient_weights = np.sum(error_tensor * self.output, axis=0)

        print(error_tensor)
        print('weight gradient', self.gradient_weights)

        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self._optimizer.calculate_update(self.bias, self.gradient_bias)
            #error_tensor_prev = self._optimizer.calculate_update(error_tensor, error_tensor_prev)

        if len(self.input_tensor.shape) > 2:
            error_tensor_prev = self.reformat(error_tensor_prev)

        #error_tensor_prev = error_tensor_prev / error_tensor
        return error_tensor_prev



    def initialize(self, weights_initializer, bias_initializer):
        self.bias = bias_initializer.initialize([self.channels])
        self.weights = weights_initializer.initialize([self.channels], self.channels, self.channels)
        return self.weights, self.bias

    def reformat(self, tensor):
        if len(tensor.shape) > 2:
            print(tensor.shape)
            B, H, M,N = tensor.shape
            # 1
            tensor = tensor.reshape(B, H, M*N)
            print(tensor.shape)
            # 2
            output_t = list(range(len(tensor)))
            for i in range(len(tensor)):
                output_t[i] = tensor[i].T
            output_t = np.asarray(output_t)
            print(output_t.shape)
            # 3
            B, MN, H = output_t.shape
            output_t= output_t.reshape(B * MN, H)
            print(output_t.shape)

        else:
            print(tensor.shape)
            B = len(self.input_tensor)
            R, H = tensor.shape
            tensor = tensor.reshape(B, int(R/B), H)
            print(tensor.shape)
            output_t = list(range(len(tensor)))
            for i in range(len(tensor)):
                output_t[i] = tensor[i].T
            output_t = np.asarray(output_t)
            print(output_t.shape)
            M = self.input_tensor.shape[2]
            output_t = output_t.reshape(B, H, M, int(R/(B*M)))

        return output_t