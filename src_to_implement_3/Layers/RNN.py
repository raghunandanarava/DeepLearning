import numpy as np
from Base import BaseLayer
import Sigmoid
import TanH
import FullyConnected


class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.memorize = False
        self.hidden_state = np.zeros(self.hidden_size)

        self.tanh = TanH.TanH()
        self.sigmoid = Sigmoid.Sigmoid()
        self.optimizer = None
        self.grad_wt_1 = None
        self.grad_wt_2 = None
        self.FC_1 = FullyConnected.FullyConnected(self.hidden_size + self.input_size, self.hidden_size)
        self.FC_2 = FullyConnected.FullyConnected(self.hidden_size, self.output_size)

        self.FC1_cache = []
        self.FC2_cache = []
        self.Sig_cache = []
        self.TanH_cache = []

        self.batch_size = None
        self.output_t = None

    def forward(self, input_tensor):
        self.batch_size = len(input_tensor)
        self.output_t = np.zeros((self.batch_size, self.output_size))
        if not self.memorize:
            self.hidden_state = np.zeros(self.hidden_size)

        for i in range(len(input_tensor)):
            new_input = np.concatenate((self.hidden_state.reshape(self.hidden_size, 1), input_tensor[i].reshape(input_tensor.shape[1], 1)))
            FC_1_f = self.FC_1.forward(new_input.T)
            self.FC1_cache.append(self.FC_1.input_tensor)

            self.hidden_state = self.tanh.forward(FC_1_f)
            self.TanH_cache.append(self.tanh.activation)

            FC_2_f = self.FC_2.forward(self.hidden_state)
            self.FC2_cache.append(self.FC_2.input_tensor)

            self.output_t[i,:] = self.sigmoid.forward(FC_2_f)
            self.Sig_cache.append(self.sigmoid.activation)

        return self.output_t

    def backward(self, error_tensor):
        prev_error_tensor = np.zeros((self.batch_size, self.input_size))
        hidden_number = np.zeros(self.hidden_size)
        self.grad_wt_1 = 0
        self.grad_wt_2 = 0

        for i in range(self.batch_size-1, -1, -1):
            self.sigmoid.activation = self.Sig_cache[i]
            temp_1 = self.sigmoid.backward(error_tensor[i])

            self.FC_2.input_tensor = self.FC2_cache[i]
            fc_2 = self.FC_2.backward(temp_1)
            self.grad_wt_2 += self.FC_2.gradient_weights

            sum = fc_2 + hidden_number

            self.tanh.activation = self.TanH_cache[i]
            fc_1 = self.tanh.backward(sum)

            self.FC_1.input_tensor = self.FC1_cache[i]
            temp_4 = self.FC_1.backward(fc_1)
            self.grad_wt_1 += self.FC_1.gradient_weights

            prev_error_tensor[i,:] = np.squeeze(np.split(temp_4.T, [self.hidden_size])[1])
            hidden_number = np.squeeze(temp_4.T[0:self.hidden_size])

        self.grad_wt_1 = np.asarray(self.grad_wt_1)
        self.grad_wt_2 = np.asarray(self.grad_wt_2)

        self.weights = self.FC_1.weights

        if self.optimizer is not None:
            self.FC_1.weights = self.optimizer.calculate_update(self.FC_1.weights, self.grad_wt_1)
            self.FC_2.weights = self.optimizer.calculate_update(self.FC_2.weights, self.grad_wt_2)

        return prev_error_tensor

    @property
    def gradient_weights(self):
        return self.grad_wt_1

    def get_weights(self):
        return self.FC_1.weights

    def set_weights(self, weights):
        self.FC_1.weights = weights

    weights = property(get_weights, set_weights)

    def get_op(self):
        return self._optimizer

    def set_op(self, op):
        self._optimizer = op

    optimizer = property(get_op, set_op)

    def initialize(self, weights_initializer, bias_initializer):
        if weights_initializer is not None and bias_initializer is not None:
            self.FC_1.initialize(weights_initializer, bias_initializer)
            self.FC_2.initialize(weights_initializer, bias_initializer)

    def set_memorize(self, value):
        self._memorize = value

    def get_memorize(self):
        return self._memorize

    memorize = property(get_memorize, set_memorize)


