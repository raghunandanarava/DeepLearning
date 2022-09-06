import numpy as np
from src_to_implement.src_to_implement.Optimization import Optimizers

class FullyConnected:

    @property

    def gradient_weights(self):
        return self._gradient_weights

    def __init__(self, input_size, output_size):
        self.input = input_size + 1
        self.output = output_size
        #the transpose of weights indeed
        self.weights = np.random.uniform(0, 1, self.output*self.input).reshape(self.input, self.output)
        print(self.weights.shape)
        self._optimizer = None

    def forward(self, input_tensor):

        # output as input for next layer
        self.input_tensor = np.append(input_tensor, np.ones((len(input_tensor), 1)), axis=1)
              #print(input.shape)

        output_tensor = np.matmul(self.input_tensor, self.weights)
        return output_tensor

    def get_op(self):
        return self._optimizer

    def set_op(self, op):
        self._optimizer = op

    optimizer = property(get_op, set_op)

    def backward(self, error_tensor):
        err_t_prev = np.matmul(self.weights[:-1, :], np.transpose(error_tensor))
        self._gradient_weights = np.matmul(np.transpose(self.input_tensor), error_tensor)
        if self._optimizer is not None:
           self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        return np.transpose(err_t_prev)

    def get_weights(self):
        return self._gradient_weights

    gradient_weights = property(get_weights)

if __name__ == "__main__":
    one = FullyConnected(4, 3)
    #print(np.random.rand(9, 4).shape)
    a = one.forward(np.random.rand(9, 4))
    print(a.shape)
    b = one.backward(a)
    print(b.shape)

    # input_tensor = np.zeros((1, 100000))
    # layer = FullyConnected(100000, 1)
    # result = layer.forward(input_tensor)
    # print(np.sum(result))
