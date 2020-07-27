import numpy as np

class MaxPool2:

    def iterate_regions(self, image):

        _, h, w, _ = image.shape
        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):
                img_region = image[:, (i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield img_region, i, j

    def forward(self, input):

        self.last_input = input
        n, h, w, num_filters = input.shape
        output = np.zeros((n, h // 2, w // 2, num_filters))

        for img_region, i, j in self.iterate_regions(input):
            for k in range(n):
                output[k, i, j] = np.amax(img_region[k], axis=(0, 1))

        return output

    def backprop(self, d_L_d_out):

        d_L_d_input = np.zeros(self.last_input.shape)

        for img_region, i, j in self.iterate_regions(self.last_input):

            n, h, w, f = img_region.shape
            amax = np.zeros((n, f))
            for x in range(n):
                amax[x] = np.amax(img_region[x], axis=(0, 1))

            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        for k in range(n):
                            if img_region[k, i2, j2, f2] == amax[k, f2]:
                                d_L_d_input[k, i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[k, i, j, f2]

        return d_L_d_input
