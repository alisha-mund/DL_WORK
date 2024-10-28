import numpy as np

from Layers.Base import BaseLayer

class TanH(BaseLayer):
    def __init__(self):
        self.trainable = False
        self.activated_input_tensor = None

    def forward(self, input_tensor):
        self.activated_input_tensor = np.tanh(input_tensor)
        return self.activated_input_tensor.copy()

    def backward(self, error_tensor):
        back_error = error_tensor * (1 - np.multiply(self.activated_input_tensor,self.activated_input_tensor))

        return back_error.copy()