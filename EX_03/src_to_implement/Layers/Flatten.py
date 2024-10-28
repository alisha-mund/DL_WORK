import numpy as np

from Layers.Base import BaseLayer


class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()
        self.flattened_arr = None
        self.input_shape = None
        self.trainable = False

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        self.flattened_arr = input_tensor.reshape(self.input_shape[0], -1)

        return self.flattened_arr

    def backward(self, error_tensor):
        backward_input = np.reshape(error_tensor, self.input_shape)

        return backward_input
