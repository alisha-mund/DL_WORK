import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.label_tensor = None
        self.predicted_tensor = None
        self.loss = None

    # Forward pass calculates the loss of the last Layer
    def forward(self, predicted_tensor, label_tensor):
        self.predicted_tensor = predicted_tensor
        self.label_tensor = label_tensor
        self.loss = np.sum(-(self.label_tensor * np.log(self.predicted_tensor + np.finfo(float).eps)))

        return self.loss

    # Backward pass calculates the error from the last layer and needs to be propagated backwards.
    def backward(self, label_tensor):
        gradient = -(label_tensor / self.predicted_tensor+np.finfo(float).eps)
        return gradient
