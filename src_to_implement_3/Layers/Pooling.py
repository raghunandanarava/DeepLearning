import numpy as np
from Base import *

class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.max_location = None
        self.input_shape = None

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        h_out = int(((len(input_tensor[1][0]) - self.pooling_shape[0]) // self.stride_shape[0]) + 1)
        w_out = int(((len(input_tensor[1][0][0]) - self.pooling_shape[1]) // self.stride_shape[1]) + 1)
        # print(h_out)
        # print(w_out)
        output_tensor = np.zeros((len(input_tensor), len(input_tensor[0]), h_out, w_out))
        self.max_location = np.zeros((len(input_tensor), len(input_tensor[0]), h_out, w_out))
        # every batch
        for i in range(len(input_tensor)):
            # every channel
            for j in range(len(input_tensor[0])):
                for k in range(0, h_out):
                    for m in range(0, w_out):
                        # vert = 0:self.pooling_shape[0]:self.stride_shape[0]
                        # hori=
                        output_tensor[i, j, k, m] = np.max(input_tensor[i, j,
                                                           k * self.stride_shape[0]:k * self.stride_shape[0] +
                                                                                    self.pooling_shape[0],
                                                           m * self.stride_shape[1]:m * self.stride_shape[1] +
                                                                                    self.pooling_shape[1]])
                        self.max_location[i, j, k, m] = np.argmax(input_tensor[i, j,
                                                                  k * self.stride_shape[0]:k * self.stride_shape[0] +
                                                                                           self.pooling_shape[0],
                                                                  m * self.stride_shape[1]:m * self.stride_shape[1] +
                                                                                           self.pooling_shape[1]])
        # print(output_tensor)
        #print(self.max_location)
        print(input_tensor.shape)
        return output_tensor

    def backward(self, error_tensor):
        #h_in = int(self.stride_shape[0] * (len(error_tensor[1][0]) - 1) + self.pooling_shape[0])
        #w_in = int(self.stride_shape[1] * (len(error_tensor[1][0][0]) - 1) + self.pooling_shape[1])
        error_tensor_prev = np.zeros(self.input_shape)
        print('error prev shape is {}'.format(error_tensor_prev.shape))
        print('error shape is {}'.format(error_tensor.shape))
        #print(len(error_tensor[1][0]))
        for i in range(len(error_tensor)):
            for j in range(len(error_tensor[0])):
                for k in range(len(error_tensor[1][0])):
                    for m in range(len(error_tensor[1][0][0])):
                        # print(self.max_location[i, j, k, m])
                        idx0 = int(self.max_location[i, j, k, m] // self.pooling_shape[0])
                        idx00 = int(self.max_location[i, j, k, m] % self.pooling_shape[1])

                        error_tensor_prev[i, j,
                        k * self.stride_shape[0]:k * self.stride_shape[0] +
                                                 self.pooling_shape[0],
                        m * self.stride_shape[1]:m * self.stride_shape[1] +
                                                 self.pooling_shape[1]][idx0, idx00] += error_tensor[i, j, k, m]


        #print(error_tensor_prev)
        return error_tensor_prev


if __name__ == "__main__":
    batch_size = 2
    input_shape = (1, 4, 4)
    np.random.seed(1337)
    # print(len(input_tensor[1][0][0]))
    input_tensor = np.array(range(np.prod(input_shape) * batch_size), dtype=np.float)
    input_tensor = input_tensor.reshape(batch_size, *input_shape)
    print(input_tensor)
    layer = Pooling((2, 2), (2, 2))
    result = layer.forward(input_tensor)
    print(result)
    # print(input_tensor.shape)
    expected_shape = np.array([batch_size, 2, 2, 3])
    expected_result = np.array([[[[5., 7.], [13., 15.]]], [[[21., 23.], [29., 31.]]]])
    err = layer.backward(result)
