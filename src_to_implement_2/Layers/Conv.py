import numpy as np
# from scipy.ndimage import convolve
from scipy.signal import convolve, correlate
import copy


class Conv:
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.stride_shape = stride_shape
        self.conv_shape = convolution_shape
        self.num_kernels = num_kernels
        self.weights = np.random.rand(self.num_kernels, *self.conv_shape)
        self.bias = np.random.rand(self.num_kernels)
        self.optimizer = None
        self.optimizer_weights = None
        self.optimizer_bias = None
        self.input_shape = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.input_shape = input_tensor.shape
        output_shape = np.asarray((len(input_tensor), self.num_kernels, *input_tensor.shape[2:]))
        output_shape[2::] = (output_shape[2::] - 1) // np.asarray(self.stride_shape) + 1

        print("input shape")
        print(self.input_shape)

        print("output shape")
        output_tensor = np.zeros(output_shape.astype(int))
        print(output_tensor.shape)

        # print(self.conv_shape)
        # print("input shape")
        # print(input_tensor.shape)

        # print("stride shape")
        # print(self.stride_shape)

        print("all shape")
        print(len(output_tensor), self.num_kernels, self.conv_shape[0])
        for i in range(len(output_tensor)):
            for j in range(self.num_kernels):
                for k in range(self.conv_shape[0]):
                    if len(self.stride_shape) == 2:
                        #print(self.weights[j, k, ::])
                        output_tensor[i, j, ::] += correlate(input_tensor[i, k, ::], self.weights[j, k, ::],
                                                             mode='same')[
                                                   ::self.stride_shape[0], ::self.stride_shape[1]]
                    elif len(self.stride_shape) == 1:
                        output_tensor[i, j, ::] += correlate(input_tensor[i, k, ::], self.weights[j, k, ::],
                                                             mode='same')[
                                                   ::self.stride_shape[0]]
                output_tensor[i, j, ::] += self.bias[j]
        # print("output shape")
        # print(output_tensor.shape)

        return output_tensor

    def backward(self, error_tensor):
        self.error_tensor = error_tensor
        error_tensor_prev = np.zeros(self.input_shape)
        print(error_tensor.shape)
        print(error_tensor_prev.shape)
        # error_tensor_prev[2::] = (error_tensor[2::] - 1) * np.asarray(self.stride_shape) + 1

        inv_weights = (np.swapaxes(self.weights, 0, 1))

        print(self.stride_shape[0])
        for i in range(len(error_tensor_prev)):
            for j in range(inv_weights.shape[0]):
                for k in range(inv_weights.shape[1]):
                    upsampled_error_tensor = np.zeros_like(error_tensor_prev[i, j, ::])
                    if len(self.stride_shape) == 2:
                        upsampled_error_tensor[::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor[i, k, ::]
                        error_tensor_prev[i, j, ::] += convolve(
                            upsampled_error_tensor, inv_weights[j, k, ::], mode='same')
                    elif len(self.stride_shape) == 1:
                        upsampled_error_tensor[::self.stride_shape[0]] = error_tensor[i, k, ::]
                        error_tensor_prev[i, j, ::] += convolve(upsampled_error_tensor,
                                                                                    inv_weights[j, k, ::],
                                                                                    mode='same')
                    # if len(self.stride_shape) == 2:
                    #     upsampled_error_tensor[i, k, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor[i, k, ::]
                    #     error_tensor_prev[i, j, ::] += convolve(
                    #         upsampled_error_tensor[i, k, ::], inv_weights[j, k, ::], mode='same')
                    # elif len(self.stride_shape) == 1:
                    #     upsampled_error_tensor[i, k, ::self.stride_shape[0]] = error_tensor[i, k, ::]
                    #     error_tensor_prev[i, j, ::] += convolve(upsampled_error_tensor[i, k, ::],
                    #                                             inv_weights[j, k, ::],
                    #                                             mode='same')
        # update gradient bias and gradient weights
        # update weights and bias
        if self.optimizer_weights is not None:
            self.weights = self.optimizer_weights.calculate_update(self.weights, self.gradient_weights)
        elif self.optimizer_bias is not None:
            self.bias = self.optimizer_bias.calculate_update(self.bias, self.gradient_bias)
        return error_tensor_prev

    def get_gradient_weights(self):
        gradient_weights = np.zeros_like(self.weights)
        pad_input = np.pad(self.input_tensor, ((0, 0), (0, 0), (self.conv_shape[1] // 2, (self.conv_shape[1] + 1) // 2 - 1),
                                               (self.conv_shape[2] // 2, (self.conv_shape[2] + 1) // 2 - 1)), 'constant',
                           constant_values=0)
        # print(gradient_weights.shape)
        for i in range(len(self.input_tensor)):
            for j in range(len(self.weights)):
                for k in range(self.weights.shape[1]):
                    upsampled_error_tensor = np.zeros_like(self.input_tensor[i, k, ::])
                    if len(self.stride_shape) == 2:
                        upsampled_error_tensor[::self.stride_shape[0], ::self.stride_shape[1]] = self.error_tensor[i, j, ::]
                        gradient_weights[j, k, ::] += \
                            correlate(pad_input[i, k, ::], upsampled_error_tensor, mode='valid')
                    elif len(self.stride_shape) == 1:
                        upsampled_error_tensor = np.zeros((self.stride_shape[0] * self.error_tensor.shape[2]))
                        upsampled_error_tensor[::self.stride_shape[0]] = self.error_tensor[i, j,::]
                        gradient_weights[j, k, ::] += \
                            correlate(pad_input[i, k, ::], upsampled_error_tensor, mode='valid')

        return gradient_weights

    def get_gradient_bias(self):
        gradient_bias = np.zeros_like(self.bias)
        for i in range(len(self.bias)):
            gradient_bias[i] = np.sum(self.error_tensor[:, i, :, :])
        return gradient_bias

    def get_op(self):
        return self.optimizer_weights, self.optimizer_bias

    def set_op(self, optimizer):
        self.optimizer_weights = copy.deepcopy(optimizer)
        self.optimizer_bias = copy.deepcopy(optimizer)

    optimizer = property(get_op, set_op)
    gradient_weights = property(get_gradient_weights)
    gradient_bias = property(get_gradient_bias)

    def initialize(self, weights_initializer, bias_initializer):

        self.weights = weights_initializer.initialize((self.num_kernels, *self.conv_shape), np.prod(self.conv_shape),
                                                      np.prod(self.conv_shape[1:]) * self.num_kernels)
        self.bias = bias_initializer.initialize((self.num_kernels, 1), self.num_kernels, 1)


if __name__ == "__main__":
    num_kernels = 4
    batch_size = 2
    conv = Conv([2], (3, 3), num_kernels)
    input_tensor = np.array(range(45 * batch_size), dtype=np.float)
    input_tensor = input_tensor.reshape((batch_size, 3, 15))
    output_tensor = conv.forward(input_tensor)
    error_tensor = conv.backward(output_tensor)
    print(error_tensor.shape)  # should be 2,3,15
