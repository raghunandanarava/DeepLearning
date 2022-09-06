import numpy as np


class ReLU:
    def __init__(self):
        return

    def forward(self, input_tensor):
        #this output is the input for next layer
        #output_tensor = np.zeros_like(input_tensor)
        self.input_tensor = input_tensor
        output_tensor = np.where(input_tensor >= 0, input_tensor, 0)
        return output_tensor

    def backward(self, error_tensor):
        prev = np.where(self.input_tensor > 0, error_tensor*1, 0)
        return prev



if __name__ == "__main__":
    input_size = 5
    batch_size = 10
    half_batch_size = int(batch_size / 2)
    input_tensor = np.ones([batch_size, input_size])
    input_tensor[0:half_batch_size, :] -= 2
    #print(input_tensor)

    expected_tensor = np.zeros([batch_size, input_size])
    expected_tensor[half_batch_size:batch_size, :] = 2
    # print(expected_tensor)

    layer = ReLU()
    output_tensor = layer.forward(input_tensor)
    output_tensor2 = layer.backward(output_tensor * 2)
    # print(output_tensor2)
    #if this = 0
    # print(np.sum(np.power(output_tensor2 - expected_tensor, 2)))


