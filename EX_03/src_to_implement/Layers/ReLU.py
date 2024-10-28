import numpy as np

from Layers.Base import BaseLayer


class ReLU(BaseLayer):
    def __init__(self):
        self.trainable = False
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        activated_input_tensor = np.maximum(0, input_tensor)
        return  activated_input_tensor

    def backward(self, error_tensor):
        back_error = error_tensor * (self.input_tensor > 0)

        return back_error
