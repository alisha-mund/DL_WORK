import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.prediction_tensor = None
        self.eps = np.finfo(float).eps

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor
        loss = - np.sum(label_tensor * np.log(prediction_tensor + self.eps))

        return loss


    def backward(self, label_tensor):
        error = - label_tensor/(self.prediction_tensor + self.eps)

        return error