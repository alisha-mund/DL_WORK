import numpy as np
from Layers.Base import BaseLayer


class FullyConnected(BaseLayer):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.error_tensor = None
        self.output_tensor = None
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self._optimizer = None

        self.new_input_tensor = []

        # Initialize randomised weights between 0 and 1, input_size+1 for biasCol
        self.weights = np.random.random((self.input_size + 1, self.output_size))

    def forward(self, input_tensor):
        # (batch_size, input_size)
        last_column = np.ones((input_tensor.shape[0], 1))
        self.new_input_tensor = np.concatenate((input_tensor, last_column), axis=1)
        self.output_tensor = np.dot(self.new_input_tensor, self.weights)
        return self.output_tensor

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def gradient_weights(self):
        return np.dot(self.new_input_tensor.T, self.error_tensor)

    def backward(self, error_tensor):
        self.error_tensor = error_tensor
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)
        later_error_tensor = np.dot(self.error_tensor, self.weights.T)

        # Return the error tensor without the Bias Column
        return later_error_tensor[:, :-1]
