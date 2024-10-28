import copy


class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.weights_intializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None

    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()
        for layer in self.layers:
            self.input_tensor = layer.forward(self.input_tensor)

        output = self.loss_layer.forward(self.input_tensor, self.label_tensor) 
        
        
        reg_output = 0
        for layer in self.layers:
            if layer.optimizer.regularizer:
                if hasattr(layer, 'weights'):
                    reg_output = layer.optimizer.regularizer.norm(layer.weights) + reg_output
                    
        return  output + reg_output

    def backward(self):
        error = self.loss_layer.backward(self.label_tensor)

        for layer in reversed(self.layers):
            error = layer.backward(error)

    def append_layer(self, layer):
        layer.optimizer = copy.deepcopy(self.optimizer)
        if hasattr(layer, 'initialize'):
            layer.initialize(self.weights_intializer, self.bias_initializer)
        self.layers.append(layer)
        
    def phase(self, testing_phase):
        for layer in self.layers:
            layer.testing_phase = testing_phase

    def train(self, iterations):
        self.iterations = iterations
        self.phase(False)
        for i in range(0, iterations):
            self.loss.append(self.forward())  # loss
            self.backward()  # updating weights by using given optimizer, repeats.

    def test(self, input_tensor):
        self.phase(True)
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)  # prediction
        return input_tensor

