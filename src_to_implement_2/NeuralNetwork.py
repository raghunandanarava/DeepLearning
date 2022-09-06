import numpy as py
import copy
import Layers as layers
import Optimizers
import Initializers
import FullyConnected
class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.weights = weights_initializer
        self.bias_initializer = bias_initializer

    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()
        input_tensor = self.input_tensor
        for i in range(len(self.layers)):
            input_tensor = self.layers[i].forward(input_tensor)
        loss_lay = self.loss_layer.forward(input_tensor, self.label_tensor)
        return loss_lay

    def backward(self):
        err_tensor = self.loss_layer.backward(self.label_tensor)
        for i in range(len(self.layers)):
            err_tensor = self.layers[len(self.layers)-1-i].backward(err_tensor)
        return err_tensor


    def append_trainable_layer(self, layer):
        layer.set_op(copy.deepcopy(self.optimizer))
        layer.initialize(self.weights, self.bias_initializer)
        self.layers.append(layer)

    def train(self, iterations):
        for i in range(iterations):
            losses = self.forward()
            self.backward()
            self.loss.append(losses)

    def test(self, input_tensor):
        for i in range(len(self.layers)):
            input_tensor = self.layers[i].forward(input_tensor)
        pred = layers.SoftMax.SoftMax()
        preds = pred.forward(input_tensor)


        return preds

if __name__ == "__main__":
    net = NeuralNetwork(Optimizers.Sgd(1),
                                      Initializers.Constant(0.123),
                                      Initializers.Constant(0.123))
    fcl_1 = FullyConnected.FullyConnected(1, 1)
    net.append_trainable_layer(fcl_1)
    fcl_2 = FullyConnected.FullyConnected(1, 1)
    net.append_trainable_layer(fcl_2)
    print(net.layers[0].weights)
