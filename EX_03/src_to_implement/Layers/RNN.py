import numpy as np
from Layers.FullyConnected import FullyConnected
from Layers.Sigmoid import Sigmoid
from Layers.TanH import TanH


class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self._optimizer = None
        self._gradient_weights = 0
        self.layer_storage = None
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_state = np.zeros(self.hidden_size)
        self.trainable = True
        self._memorize = False
        self.memory_states = []
        self.layers = []
        self.layers_initialization()
        
    def layers_initialization(self):
        layer_input_1 = self.input_size + self.hidden_size  # uses the current i/p and prev hidden state
        layer_input_2 = self.hidden_size  # updates the hidden states
        initial_layer = FullyConnected(layer_input_1, self.hidden_size)
        self.layers.append(initial_layer)
        self.layers.append(TanH())

        output_layer = FullyConnected(layer_input_2, self.output_size)
        self.layers.append(output_layer)

        self.layers.append(Sigmoid())

        self.layer_storage = [[] for _ in range(len(self.layers))]

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, value):
        self._memorize = value

    def forward(self, input_tensor):
        time_dimension = input_tensor.shape[0]

        if not self.memorize:
            self.hidden_state = np.zeros(self.hidden_size)

        # self.memory_states.append(self.hidden_state)

        output_tensor = np.zeros((time_dimension, self.output_size))

        for time in range(time_dimension):
            input_dim = input_tensor[time].flatten()
            hidden_dim = self.hidden_state.flatten()

            new_input = np.concatenate((hidden_dim, input_dim)).reshape(1, -1)

            out_1 = self.layers[0].forward(new_input)
            self.hidden_state = self.layers[1].forward(out_1)
            out_3 = self.layers[2].forward(self.hidden_state)
            out_4 = self.layers[3].forward(out_3)

            output_tensor[time] = out_4

            self.layer_storage[0].append(self.layers[0].next_input_tensor)
            self.layer_storage[1].append(self.layers[1].activated_input_tensor)
            self.layer_storage[2].append(self.layers[2].next_input_tensor)
            self.layer_storage[3].append(self.layers[3].activated_input_tensor)

        return output_tensor

    def backward(self, error_tensor):
        time_dimension = error_tensor.shape[0]

        final_gradients = np.zeros((time_dimension, self.input_size))

        hidden_gradients = np.zeros(self.hidden_size)
        layer_2_gradient = 0
        layer_0_gradient = 0

        for time in reversed(range(time_dimension)):
            error = error_tensor[time]

            # initialize values for back propagation
            self.layers[3].activated_input_tensor = self.layer_storage[3][time]
            self.layers[2].next_input_tensor = self.layer_storage[2][time]
            self.layers[1].activated_input_tensor = self.layer_storage[1][time]
            self.layers[0].next_input_tensor = self.layer_storage[0][time]

            error_1 = self.layers[3].backward(error)

            error_2 = self.layers[2].backward(error_1)
            error_3 = hidden_gradients + error_2
            layer_2_gradient += self.layers[2].gradient_weights

            error_4 = self.layers[1].backward(error_3)

            error_5 = self.layers[0].backward(error_4)

            layer_0_gradient += self.layers[0].gradient_weights

            transposed_error = error_5.T

            final_gradients[time] = transposed_error[self.hidden_size::].reshape(-1)

            hidden_gradients = transposed_error[0:self.hidden_size].reshape(-1)

        self._gradient_weights = layer_0_gradient

        if self._optimizer is not None:
            self.layers[0].weights = self.optimizer.calculate_update(self.layers[0].weights, layer_0_gradient)
            self.layers[2].weights = self.optimizer.calculate_update(self.layers[2].weights, layer_2_gradient)

        return final_gradients

    def initialize(self, weights_initializer, bias_initializer):
        if weights_initializer is not None and bias_initializer is not None:
            self.layers[0].initialize(weights_initializer, bias_initializer)
            self.layers[2].initialize(weights_initializer, bias_initializer)

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def weights(self):
        return self.layers[0].weights

    @weights.setter
    def weights(self, value):
        self.layers[0].weights = value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value
        return
