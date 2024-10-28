import numpy as np

from Layers.Base import BaseLayer


class SoftMax(BaseLayer):
    def __init__(self):
        self.trainable = False
        self.predicted_prob = None

    def forward(self, input_tensor):
        max_input = np.max(input_tensor)
        exp_xi = np.exp(input_tensor-max_input)
        exp_xj = np.sum(exp_xi, axis=1, keepdims=True)

        self.predicted_prob = exp_xi/exp_xj

        return self.predicted_prob

    def backward(self, error_tensor):
        back_error = self.predicted_prob * (error_tensor - np.sum(error_tensor * self.predicted_prob, axis=1, keepdims=True))

        return back_error

