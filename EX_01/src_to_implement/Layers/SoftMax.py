import numpy as np
from Layers.Base import BaseLayer


class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.predicted_probability = None
        self.input_tensor = None
        self.trainable = False  # no need of training

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        # Subtracting max input value from each input helps prevent overflow while processing large inputs
        exponents = np.exp(self.input_tensor - np.max(self.input_tensor))
        self.predicted_probability = exponents / np.sum(exponents, axis=1, keepdims=True)
        return self.predicted_probability

    def backward(self, error_tensor):
        output = self.predicted_probability * (
                    error_tensor - np.sum(error_tensor * self.predicted_probability, axis=1, keepdims=True))
        return output
