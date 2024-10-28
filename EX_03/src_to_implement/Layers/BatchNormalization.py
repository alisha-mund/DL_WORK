
from Layers.Base import BaseLayer
import numpy as np

from Layers.Helpers import compute_bn_gradients

class BatchNormalization (BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels  #no of channels for input (vector and image)
        self.weights = None
        self.bias = None
        self.initialize(0,0)
        self.trainable = True

        self.tt_mean = None
        self.tt_var = None

        self.gradient_weights = None
        self.gradient_bias = None
        self.optimizer = None


    def forward(self, input_tensor):
        self.input_tensor = input_tensor

        if input_tensor.ndim != 4:
            self.checkconv = False

        else:
            self.checkconv = True
            self.original_shape = input_tensor.shape
            input_tensor = self.reformat(input_tensor)

        self.tt_mean = np.mean(input_tensor,axis=0) if self.tt_mean is None else self.tt_mean
        self.tt_var = np.var(input_tensor,axis=0) if self.tt_var is None else self.tt_var

        if self.testing_phase:
            self.newx = (input_tensor - self.tt_mean) / (self.tt_var + 1e-11) ** 0.5

        else:
            self.tr_mean = np.mean(input_tensor,axis=0)
            self.tr_var = np.var(input_tensor,axis=0)
            self.tt_mean = 0.7 * self.tt_mean + 0.3 * self.tr_mean
            self.tt_var = 0.7 * self.tt_var + 0.3 * self.tr_var
            self.newx = (input_tensor - self.tr_mean) / (self.tr_var + 1e-11) ** 0.5

        output = self.weights * self.newx + self.bias

        return self.reformat(output) if self.checkconv else output


    def backward(self, error_tensor):

        if self.checkconv:
            error_tensor = self.reformat(error_tensor)
            self.input_tensor = self.reformat(self.input_tensor)

        output = compute_bn_gradients(error_tensor, self.input_tensor, self.weights, self.tr_mean, self.tr_var)
        output = self.reformat(output) if self.checkconv else output

        xnorinp = (self.input_tensor - self.tr_mean) / (self.tr_var + 1e-11) ** 0.5

        self.gradient_weights = np.sum(error_tensor * xnorinp,axis=0)
        self.gradient_bias = np.sum(error_tensor,axis=0)

        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self.optimizer.calculate_update(self.bias, self.gradient_bias)

        return output


    def initialize(self, _, __):
        self.weights = np.ones(self.channels) #weights
        self.bias = np.zeros(self.channels) #bias


    def reformat(self, tensor):
        if tensor.ndim == 2:
            cbatch = self.original_shape[0]
            cchannel = self.original_shape[1]
            cheight = self.original_shape[2]
            cwidth = self.original_shape[3]
            reshape_tensor = np.zeros(tensor.shape)
            reshape_tensor = (tensor.reshape(cbatch, cheight*cwidth, cchannel).transpose((0,2,1))).reshape(cbatch,cchannel,cheight,cwidth)
            return reshape_tensor

        elif tensor.ndim == 4:
            batch = tensor.shape[0]
            channel = tensor.shape[1]
            height = tensor.shape[2]
            width = tensor.shape[3]
            reshape_tensor = np.zeros(tensor.shape)
            reshape_tensor = tensor.reshape(batch,channel,height*width).transpose((0,2,1)).reshape(batch*height*width,channel)
            return reshape_tensor



@property
def gradient_weights(self):
    return self.gradient_weights

@gradient_weights.setter
def gradient_weights(self,gradient_weights):
    self.gradient_weights = gradient_weights
    return

@property
def gradient_bias(self):
    return self.gradient_bias

@gradient_bias.setter
def gradient_bias(self,gradient_bias):
    self.gradient_bias = gradient_bias
    return

@property
def optimizer(self):
    return self.optimizer

@optimizer.setter
def optimizer(self,optimizer):
    self.optimizer = optimizer
    return