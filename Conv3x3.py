import numpy as np

class Conv3x3:

    def __init__(self, num_filters):

        self.num_filters = num_filters
        self.filters = np.random.rand(num_filters, 3, 3) * (2 * 1) - 1

    def iterate_regions(self, image):

        _, h, w = image.shape

        for i in range(h - 2):
            for j in range(w - 2):
                img_region = image[:, i:(i + 3), j:(j + 3)]
                yield img_region, i, j

    def forward(self, input):

        self.last_input = input

        n, h, w = input.shape
        output = np.zeros((n, h - 2, w - 2, self.num_filters))

        for img_region, i, j in self.iterate_regions(input):
            for k in range(n):
                output[k, i, j] = np.sum(img_region[k] * self.filters, axis=(1, 2))

        return output

    def backprop(self, d_L_d_out, learn_rate):

        d_L_d_filters = np.zeros(self.filters.shape)

        for img_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                for k in range(self.last_input.shape[0]):
                    d_L_d_filters[f] += d_L_d_out[k, i, j, f] * img_region[k]

        self.filters -= learn_rate * d_L_d_filters

        return None
