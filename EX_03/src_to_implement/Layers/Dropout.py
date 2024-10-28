import numpy as np

from Layers.Base import BaseLayer

class Dropout(BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability
        self.testing_phase = False
        self. Dtmtr = None

    def forward(self, input_tensor):
        output = input_tensor  # in testing phase

        if not self.testing_phase:    # for training phase
            self.Dtmtr = np.random.rand(*input_tensor.shape) < self.probability
            output = np.multiply(output, self.Dtmtr)
            output = output / self.probability
        return output

    def backward(self, error_tensor):
        output = np.multiply(error_tensor, self.Dtmtr)
        output = output / self.probability
        return output