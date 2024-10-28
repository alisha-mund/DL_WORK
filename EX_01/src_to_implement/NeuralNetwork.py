import copy
from Layers import *


class NeuralNetwork:

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.iterations = None
        self.input_tensor = None
        self.label_tensor = None

    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()
        for layer in self.layers:
            self.input_tensor = layer.forward(self.input_tensor)

        # assigns loss to last loss_layer
        return self.loss_layer.forward(self.input_tensor, self.label_tensor)

    def backward(self):
        error = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error = layer.backward(error)

    def append_layer(self, layer):
        layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self, iterations):
        self.iterations = iterations
        for i in range(0, iterations):
            self.loss.append(self.forward())  # loss
            self.backward()  # updating weights by using given optimizer, repeats.

    def test(self, input_tensor):
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)  # prediction
        return input_tensor
