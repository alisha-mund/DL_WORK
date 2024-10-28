import numpy as np

from Layers.Base import BaseLayer
from scipy.signal import correlate
from copy import deepcopy
from math import ceil

class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels

        self.trainable = True
        self.weights = np.random.uniform(0,1,size =(self.num_kernels,) + self.convolution_shape)
        self.bias = np.random.uniform(0,1,size=(self.num_kernels,1))
        self._gradient_weights = None
        self._gradient_bias = None
        self._optimizer = None
        self._optimizer_weights = None
        self._optimizer_bias = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        NoBatch = input_tensor.shape[0]

        # checking for 2D case
        if len(self.convolution_shape) == 3:
            InputHeight = input_tensor.shape[2]
            InputWidth = input_tensor.shape[3]
            NoInputChannels =  self.convolution_shape[0]
            FilterHeight = self.convolution_shape[1]

            # 2D is defined in b, c, y, x order
            output_tensor = np.zeros((NoBatch, self.num_kernels, InputHeight, InputWidth))

            for b in range(NoBatch):
                for k in range(self.num_kernels):
                    for c in range(NoInputChannels):
                        wgts_flip = np.flip(self.weights[k,c,:,:])
                        output_tensor[b,k,:,:] = output_tensor[b,k,:,:] + correlate(input_tensor[b,c,:,:], wgts_flip, mode='same')
                    output_tensor[b,k,:,:] = output_tensor[b,k,:,:] + self.bias[k]

            # taking the required from stride info
            output_tensor = output_tensor[:,:,::self.stride_shape[0],::self.stride_shape[1]]
            return output_tensor

        # checking for 1D case
        elif len(self.convolution_shape) == 2:
            InputHeight = input_tensor.shape[2]
            NoInputChannels = self.convolution_shape[0]
            FilterHeight = self.convolution_shape[1]

            # 1D is defined in b, c, y order
            output_tensor = np.zeros((NoBatch, self.num_kernels, InputHeight))

            for b in range(NoBatch):
                for k in range(self.num_kernels):
                    for c in range(NoInputChannels):
                        wgts_flip = np.flip(self.weights[k,c,:])
                        output_tensor[b,k,:] = output_tensor[b,k,:] + correlate(input_tensor[b,c,:], wgts_flip, mode='same')
                    output_tensor[b,k,:] = output_tensor[b,k,:] + self.bias[k]

            # taking the required from stride info
            output_tensor = output_tensor[:,:,::self.stride_shape[0]]
            return output_tensor

        else:
            raise ValueError("Invalid Convolution Shape")



    def initialize(self, weights_initalizer, bias_intializer):
        self.weights = weights_initalizer.initialize((self.num_kernels,) + self.convolution_shape, np.prod(self.convolution_shape), (self.num_kernels * np.prod(self.convolution_shape))/ self.convolution_shape[0])
        self.bias = bias_intializer.initialize((self.num_kernels,1), np.prod(self.convolution_shape), (self.num_kernels * np.prod(self.convolution_shape))/ self.convolution_shape[0])



    def backward(self, error_tensor):
        NoBatch = error_tensor.shape[0]
        Nochannels = self.input_tensor.shape[1]

        # checking for 2D case
        if len(self.convolution_shape) == 3:
            NoInputChannels =  self.convolution_shape[0]
            FilterHeight = self.convolution_shape[1]
            FilterWidth = self.convolution_shape[2]

            # going to the orginal input tensor shape as error tensor is strided
            error_tensor_upscaled = np.zeros((error_tensor.shape[0], error_tensor.shape[1], self.input_tensor.shape[2], self.input_tensor.shape[3]))
            error_tensor_upscaled[:,:,::self.stride_shape[0],::self.stride_shape[1]] = error_tensor

            pad_gradient_output =  np.pad(self.input_tensor, np.array(((0,0), (0,0), (ceil((FilterHeight-1)/2), (FilterHeight-1)//2), (ceil((FilterWidth-1)/2),(FilterWidth-1)//2)), dtype="int"), mode='constant')
            gradient_output = np.zeros(self.input_tensor.shape)

            self._gradient_weights = np.zeros((self.num_kernels,) + self.convolution_shape)
            self._gradient_bias = np.zeros((self.num_kernels,1))

            for b in range(NoBatch):
                for k in range(self.num_kernels):
                    for c in range(NoInputChannels):
                        self._gradient_weights[k,c,:,:] = self._gradient_weights[k,c,:,:] + correlate(error_tensor_upscaled[b,k,:,:], pad_gradient_output[b,c,:,:], mode='valid')

            # summing up along all axes of the error tensor
            self._gradient_bias = np.sum(np.sum(np.sum(error_tensor_upscaled,axis=-1),axis=-1),axis=0).reshape(-1, 1)
            wgts_flip = np.moveaxis(self.weights, [0,1],[1,0])

            for b in range(NoBatch):
                for c in range(Nochannels):
                    for k in range(self.num_kernels):
                        gradient_output[b,c,:,:] = gradient_output[b,c,:,:] + correlate(error_tensor_upscaled[b,k,:,:], wgts_flip[c,k,:,:], mode="same")

            if self._optimizer is not None:
                self.weights = self._optimizer_weights.calculate_update(self.weights, self._gradient_weights)
                self.bias = self._optimizer_bias.calculate_update(self.bias, self._gradient_bias)
            return gradient_output

        # checking for 1D case
        elif len(self.convolution_shape) == 2:
            OutputHeight = error_tensor.shape[2]
            InputHeight = self.input_tensor.shape[2]
            NoInputChannels = self.convolution_shape[0]
            FilterHeight = self.convolution_shape[1]

            # going to the orginal input tensor shape as error tensor is strided
            error_tensor_upscaled = np.zeros((error_tensor.shape[0], error_tensor.shape[1], self.input_tensor.shape[2]))
            error_tensor_upscaled[:,:,::self.stride_shape[0]] = error_tensor

            pad_gradient_output =  np.pad(self.input_tensor, np.array(((0,0), (0,0), (ceil((FilterHeight-1)/2), (FilterHeight-1)//2)), dtype="int"), mode='constant')
            gradient_output = np.zeros(self.input_tensor.shape)

            self._gradient_weights = np.zeros((self.num_kernels,) + self.convolution_shape)
            self._gradient_bias = np.zeros((self.num_kernels,1))

            for b in range(NoBatch):
                for k in range(self.num_kernels):
                    for c in range(NoInputChannels):
                        self._gradient_weights[k,c,:] = self._gradient_weights[k,c,:] + correlate(error_tensor_upscaled[b,k,:], pad_gradient_output[b,c,:], mode='valid')

             # summing up along all axes of the error tensor
            self._gradient_bias = np.sum(np.sum(np.sum(error_tensor_upscaled,axis=-1),axis=-1),axis=0).reshape(-1, 1)
            wgts_flip = np.moveaxis(self.weights, [0,1],[1,0])

            for b in range(NoBatch):
                for c in range(Nochannels):
                    for k in range(self.num_kernels):
                        gradient_output[b,c,:] = gradient_output[b,c,:] + correlate(error_tensor_upscaled[b,k,:], wgts_flip[c,k,:], mode="same")

            if self._optimizer is not None:
                self.weights = self._optimizer_weights.calculate_update(self.weights, self._gradient_weights)
                self.bias = self._optimizer_bias.calculate_update(self.bias, self._gradient_bias)
            return gradient_output

        else:
            raise ValueError("Invalid Convolution Shape")


    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizer_weights = deepcopy(optimizer)
        self._optimizer_bias = deepcopy(optimizer)