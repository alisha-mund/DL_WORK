import numpy as np

from Layers.Base import BaseLayer


class Constant(BaseLayer):
    def __init__(self, default_val=0.1):
        self.const_in = default_val

    def initialize(self, weight_shape, fan_in, fan_out):
        weight_initializer = np.ones(weight_shape) * self.const_in

        return weight_initializer

class UniformRandom(BaseLayer):
    def __init__(self):
        self.weight_initializer = None

    def initialize(self, weight_shape, fan_in, fan_out):
        if len(weight_shape) == 1:
            self.weight_initializer = np.random.rand(weight_shape[0])
        else:
            self.weight_initializer = np.random.rand(weight_shape[0], weight_shape[1])
        
        return self.weight_initializer

class Xavier(BaseLayer):
    def __init__(self):
        self.weight_initializer = None
        self.standard_dev = None

    def initialize(self, weight_shape, fan_in, fan_out):
        self.standard_dev = np.sqrt(2/(fan_in + fan_out))

        self.weight_initializer = np.random.normal(loc=0, scale=self.standard_dev, size=weight_shape)

        return self.weight_initializer

class He(BaseLayer):
    def __init__(self):
        self.weight_initializer = None
        self.standard_dev = None

    def initialize(self, weight_shape, fan_in, fan_out):
        self.standard_dev = np.sqrt(2/fan_in)

        self.weight_initializer = np.random.normal(loc=0, scale=self.standard_dev, size=weight_shape)

        return self.weight_initializer
