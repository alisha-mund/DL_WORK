import numpy as np
from Layers.Base import BaseLayer


class ReLU(BaseLayer):

    def __init__(self):
        super().__init__()
        self.input = None
        self.trainable = False

    def forward(self, input_tensor):
        self.input = input_tensor
        tensor = np.maximum(0, input_tensor)
        return tensor

    def backward(self, error_tensor):
        return np.where(self.input > 0, error_tensor, 0)
