import numpy as np


class Pooling():
    def __init__(self, stride_shape, pooling_shape):
        self.max_pooled_locations = None
        self.trainable = False
        self.stride_shape = stride_shape
        self.stride_height, self.stride_width = stride_shape
        self.pooling_shape = pooling_shape
        self.pool_height, self.pool_width = pooling_shape

        self.output_tensor = None
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor

        if len(self.input_tensor.shape) == 4:
            # 1D is defined in b, c, y order, for 2D in b, c, y, x order.
            batch, channels, in_height, in_width = self.input_tensor.shape

            # Output Dimensions:  batch * channels * [((y - f)/s) + 1] * [((x - f)/s) + 1]
            out_width = ((in_width - self.pool_width) // self.stride_width) + 1
            out_height = ((in_height - self.pool_height) // self.stride_height) + 1

            self.output_tensor = np.zeros((batch, channels, out_height, out_width))
            self.max_pooled_locations = np.zeros_like(self.input_tensor)

            # Making sure the pooling size and stride size are same
            if self.stride_shape[1] == self.pooling_shape[1]:
                for b in range(batch):
                    for c in range(channels):
                        for h in range(out_height):
                            for x in range(out_width):
                                x_1 = x * self.stride_width
                                h_1 = h * self.stride_height
                                # Taking the window to focus on for pooling
                                max_window = self.input_tensor[b, c, h_1:h_1 + self.pool_height, x_1:x_1 + self.pool_width]
                                pool_max = np.max(max_window)

                                self.output_tensor[b, c, h, x] = pool_max

                                # Keeping track of the indices where the maximum value or values are taken from to be used in backward pass
                                mask = (max_window == pool_max)
                                self.max_pooled_locations[b, c, h_1:h_1 + self.pool_height, x_1:x_1 + self.pool_width] = mask

            return self.output_tensor

    def backward(self, error_tensor):
        batch, channels, in_height, in_width = self.input_tensor.shape

        output_height = self.output_tensor.shape[2]
        output_width = self.output_tensor.shape[3]

        # Gradient tensor as same shape as input tensor
        self.backward_output = np.zeros_like(self.input_tensor)

        # Making sure the pooling size and stride size are same
        if self.stride_shape[1] == self.pooling_shape[1]:
            for b in range(batch):
                for c in range(channels):
                    for h in range(output_height):
                        for x in range(output_width):
                            x_1 = x * self.stride_width
                            h_1 = h * self.stride_height

                            # mask with boolean values indicating the location of maximum values
                            mask = self.max_pooled_locations[b, c, h_1:h_1 + self.pool_height, x_1:x_1 + self.pool_width]

                            # error multiplied with boolean mask: Propagates error only to the maximum value locations
                            self.backward_output[b, c, h_1:h_1 + self.pool_height, x_1:x_1 + self.pool_width] += mask * error_tensor[b, c, h, x]

        return self.backward_output
