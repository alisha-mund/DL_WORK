import numpy as np

class Optimizer:
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer


class Sgd(Optimizer):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):

        # Calculate gradient weights using basic grad descent formula
        gradient_update = self.learning_rate * gradient_tensor
        new_weights = weight_tensor - gradient_update # when self.regularizer is None
        
        if self.regularizer:
            new_weights = weight_tensor - gradient_update - ( self.learning_rate * self.regularizer.calculate_gradient(weight_tensor))
        return new_weights


class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = momentum_rate
        self.velocity_k = 0
        self.weight_next = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.velocity_k = self.mu * self.velocity_k - self.learning_rate * gradient_tensor

        self.weight_next = weight_tensor + self.velocity_k
        
        if self.regularizer:
            self.weight_next = self.weight_next - (self.learning_rate * self.regularizer.calculate_gradient(weight_tensor))

        return self.weight_next


class Adam(Optimizer):
    def __init__(self, learning_rate, mu, rho):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.velocity = 0
        self.rk = 0
        self.k_pow = 1
        self.weight_tensor = None
        self.eps = 1e-8

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.velocity = self.mu * self.velocity + (1 - self.mu) * gradient_tensor

        self.rk = self.rho * self.rk + (1 - self.rho) * (gradient_tensor * gradient_tensor)

        # Bias Correction term
        velocity_hat = self.velocity / (1 - (self.mu ** self.k_pow))
        rk_hat = self.rk / (1 - (self.rho ** self.k_pow))

        self.weight_tensor = weight_tensor - self.learning_rate * (velocity_hat / (np.sqrt(rk_hat) + self.eps))

        self.k_pow += 1
        
        if self.regularizer:
            self.weight_tensor = self.weight_tensor - (self.learning_rate * self.regularizer.calculate_gradient(weight_tensor))
        return self.weight_tensor
