from Layers import *
from Optimization import *
import copy


class NeuralNetwork:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = list()
        self.layers = list()
        self.data_layer = None
        self.loss_layer = None
        self.labelling = None

    def forward(self):
        ip, self.labelling = self.data_layer.next()
        for j in self.layers:
            ip = j.forward(ip)
        return self.loss_layer.forward(ip, self.labelling)

    def backward(self):
        error = self.loss_layer.backward(self.labelling)
        for i in self.layers[::-1]:
            error = i.backward(error)

    def append_trainable_layer(self, layer):
        self.optimizer = copy.deepcopy(self.optimizer)
        layer.optimizer = self.optimizer
        self.layers.append(layer)

    def train(self, iterations):
        for it in range(iterations):
            self.loss.append(self.forward())
            self.backward()

    def test(self, input_tensor):
        for la in self.layers:
            input_tensor = la.forward(input_tensor)
        return input_tensor
