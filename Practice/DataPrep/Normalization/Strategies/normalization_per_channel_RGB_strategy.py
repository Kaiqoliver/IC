import numpy as np

#Every class in this file has some form of normalization method

#This normalization class receives a numpy data in RGB format, that is:  (number_of_images, height, width, 3),
#and measures the mean and variance of each channel in each image to apply the normalization individually for
#each channel in that same image.
class NormalizationPerChannelRGBStrategy():
    def normalize(self, data, height, width):
        data_mean_per_channel = np.mean(data, axis=(1, 2), keepdims=True)
        data_variance_per_channel = np.var(data, axis=(1,2), keepdims=True)

        normalized_data = (data - data_mean_per_channel) / np.sqrt(data_variance_per_channel)

        return normalized_data
