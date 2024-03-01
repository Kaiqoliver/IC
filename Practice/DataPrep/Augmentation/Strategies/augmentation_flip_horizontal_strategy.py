import numpy as np

#This strategy flips de data horizontally and then concatenate it to the original data
class AugmentationFlipHorizontal:
    def augment(self, data):
        flipped_data = np.empty_like(data)

        for i in range(data.shape[0]):
            flipped_data[i] = np.fliplr(data[i])

        augmented_data = np.concatenate([data, flipped_data], axis=0)

        return augmented_data