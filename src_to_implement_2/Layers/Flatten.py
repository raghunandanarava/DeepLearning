import numpy as np

class Flatten:
    def __init__(self):
        return

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        output_tensor = list()
        for i in range(len(self.input_tensor)):
            output_tensor.append(input_tensor[i].flatten())
        output_tensor = np.asarray(output_tensor)

        return output_tensor

    def backward(self, error_tensor):
        prev_tensor = error_tensor.reshape(len(error_tensor), *self.input_tensor[0].shape)
        # for i in range(len(error_tensor)):
        #     prev_tensor.append(error_tensor.reshape())
        # output_tensor = np.asarray(output_tensor)
        return prev_tensor

if __name__ == "__main__":
    batch_size = 9
    input_shape = (3, 4, 11)
    input_tensor = np.array(range(int(np.prod(input_shape) * batch_size)),
                            dtype=np.float).reshape(batch_size, *input_shape)

    #input_tensor =  (batch_size, input_shape))

    print(input_tensor.shape)

    flatten = Flatten()
    output_tensor = flatten.forward(input_tensor)
    print(output_tensor.shape)
    backward_tensor = flatten.backward(output_tensor)


    output_tensor = flatten.forward(input_tensor)
    input_vector = np.array(range(int(np.prod(input_shape) * batch_size)), dtype=np.float)
    input_vector = input_vector.reshape(batch_size, np.prod(input_shape))