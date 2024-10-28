class Sgd:

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.gradient_tensor = None
        self.weight_tensor = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.weight_tensor = weight_tensor
        self.gradient_tensor = gradient_tensor

        updated_tensor = weight_tensor - self.learning_rate * gradient_tensor
        return updated_tensor

