import numpy as np

class L1_Regularizer():

    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights): #calculate sud-gradient
        subgrad = self.alpha * np.sign(weights)
        return subgrad

    def norm(self, weights):   #calculate norm
        calnorm = self.alpha * np.sum(abs(weights))
        return calnorm



class L2_Regularizer():

    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        subgrad = self.alpha * weights
        return subgrad

    def norm(self,weights):
        calnorm = self.alpha * np.sum(np.multiply(weights,weights))
        return calnorm