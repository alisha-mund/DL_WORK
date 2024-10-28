import numpy as np


class Sgd:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):

        # Calculate gradient weights using basic grad descent formula
        gradient_update = self.learning_rate * gradient_tensor
        new_weights = weight_tensor - gradient_update
        return new_weights


class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.mu = momentum_rate
        self.velocity_k = 0
        self.weight_next = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.velocity_k = self.mu * self.velocity_k - self.learning_rate * gradient_tensor

        self.weight_next = weight_tensor + self.velocity_k

        return self.weight_next


class Adam:
    def __init__(self, learning_rate, mu, rho):
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
        return self.weight_tensor
