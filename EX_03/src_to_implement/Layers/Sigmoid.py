import numpy as np

from Layers.Base import BaseLayer


class Sigmoid(BaseLayer):
    def __init__(self):
        self.trainable = False
        self.activated_input_tensor = None

    def forward(self, input_tensor):
        self.activated_input_tensor = 1 / (1 + np.exp(-input_tensor))
        return self.activated_input_tensor.copy()

    def backward(self, error_tensor):
        back_error = error_tensor * np.multiply(self.activated_input_tensor, 1 - self.activated_input_tensor)

        return back_error.copy()
