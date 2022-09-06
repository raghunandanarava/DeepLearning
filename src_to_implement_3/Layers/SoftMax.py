import numpy as np


class SoftMax:

    def __init__(self):
        return

    def forward(self, input_tensor):
        self.pred = np.zeros_like(input_tensor)
        print(np.max(input_tensor).shape)
        x_k = np.exp(input_tensor - np.max(input_tensor))
        print(x_k.shape)
        self.pred = np.transpose(np.transpose(x_k) / np.transpose(np.sum(x_k, axis=1)))
        return self.pred


    def backward(self, error_tensor):
        diff = np.transpose(np.sum(np.multiply(error_tensor, self.pred), axis=1))
        error_tensor_prev = np.multiply(self.pred, np.transpose(np.transpose(error_tensor) - diff))
        return error_tensor_prev


if __name__ == "__main__":
    batch_size = 9
    categories = 4

    label_tensor = np.zeros([batch_size, categories])
    for i in range(batch_size):
        label_tensor[i, np.random.randint(0, categories)] = 1

    #input_tensor = np.zeros([batch_size, categories]) + 10000.
    #print(input_tensor)
    #print(len(input_tensor))
    #layer = SoftMax()

    #not nan
    #pred = layer.forward(input_tensor)
    #print(pred)

    # input_tensor = np.arange(categories * batch_size)
    # input_tensor = input_tensor / 100.
    # input_tensor = input_tensor.reshape((categories, batch_size))
    # print(input_tensor)
    # layer = SoftMax()
    # prediction = layer.forward(input_tensor.T)
    # print(input_tensor.T)
    # print(prediction)
    # print(prediction)
    # input_tensor = self.label_tensor * 100.
    # layer = SoftMax.SoftMax()
    # loss_layer = L2Loss()
    # pred = layer.forward(input_tensor)
    # #loss must < 1e-10
    # loss = loss_layer.forward(pred, self.label_tensor)
    input_tensor = label_tensor * 100.
    layer = SoftMax()
    #loss_layer = CrossEntropyLoss()
    pred = layer.forward(input_tensor)
    #loss_layer.forward(pred, self.label_tensor)
    #error = loss_layer.backward(self.label_tensor)
    #error = layer.backward(error)