import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        return

    def forward(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        losses = np.where(label_tensor == 1, - np.log(self.input_tensor + np.finfo(float).eps), self.input_tensor)
        loss = np.sum(losses[label_tensor == 1])
        print(losses)
        # losses = np.zeros(len(loss))
        # for i in range(len(loss)):
        #     losses = np.sum(loss[i, :])
        return loss

    def backward(self, label_tensor):
        err_prev = - label_tensor / self.input_tensor
        return err_prev



if __name__ == "__main__":
    batch_size = 9
    categories = 4
    label_tensor0= np.zeros([batch_size, categories])
    for i in range(batch_size):
        label_tensor0[i, np.random.randint(0, categories)] = 1
    print(label_tensor0)
    input_tensor0 = np.abs(np.random.random(label_tensor0.shape))
    #print(input_tensor0)
    layer = CrossEntropyLoss()
    loss0 = layer.forward(label_tensor0, label_tensor0)
    print(loss0)
    #assertAlmostEqual(loss, 0)

    label_tensor = np.zeros((batch_size, categories))
    label_tensor[:, 2] = 1
    print(label_tensor)
    input_tensor = np.zeros_like(label_tensor)
    input_tensor[:, 1] = 1
    #print(input_tensor)
    layer = CrossEntropyLoss()
    loss = layer.forward(input_tensor, label_tensor) #324.3928805
    print(loss)

    #to test gradient
    input_tensor2 = np.abs(np.random.random(label_tensor.shape))
    layers = list()
    layers.append(CrossEntropyLoss())