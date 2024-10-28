from Layers.Base import BaseLayer
import numpy as np


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        self._gradient_weight = None
        self.next_error = None
        self.output = None
        self.trainable = True

        self.input_size = input_size
        self.output_size = output_size
        # Initialize random weights between 0 and 1, input_size+1 for biasCol
        self.weights = np.random.random((input_size + 1, output_size))

        self.next_input_tensor = None
        self._optimizer = None

    def initialize(self, weights_initializer, bias_initializer):
        self.weights[:-1] = weights_initializer.initialize((self.input_size, self.output_size), self.input_size, self.output_size)
        self.weights[-1] = bias_initializer.initialize((self.output_size,), self.input_size, self.output_size)

    def forward(self, input_tensor):  # input_tensor matrix = input size columns * batch size rows.
        bias_col = np.ones((input_tensor.shape[0], 1))
        self.next_input_tensor = np.append(input_tensor, bias_col, axis=1)
        self.output = np.dot(self.next_input_tensor, self.weights)
        return self.output

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    def backward(self, error_tensor):
        self._gradient_weight = np.dot(self.next_input_tensor.T, error_tensor)

        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self._gradient_weight)

        next_error = np.dot(error_tensor, self.weights.T)
        next_error = next_error[:, :-1]

        return next_error

    @property
    def gradient_weights(self):
        return self._gradient_weight
