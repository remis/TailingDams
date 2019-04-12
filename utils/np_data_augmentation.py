"""
data augmentation functions on 'images' as numpy arrays
"""

import numpy as np


def random_flip(image_array, axis=None, prob=0.5):
    """
    randomly flips an input image
    :param image_array: input image as a numpy array [height, width, channels]
    :param axis: along which axis to flip, if None then 0 or 1 are randomly chosen
    :param prob: with the (1-prob) probability the unaltered image is returned
    :return: flipped image of the same size
    """
    p = np.random.rand(1)[0]
    if p < prob:
        # flip the image
        if axis is None:
            axis = np.random.choice(2)
        return np.flip(image_array, axis=axis)
    else:
        return image_array


def random_rotation(image_array, prob=0.5, angles=[90, 180, 270]):
    """
    randomly rotate an input image by 90, 180 or 270 degrees
    :param image_array: input image as a numpy array [height, width, channels]
    :param prob: with the (1-prob) probability the unaltered image is returned
    :param angles: possible angles to choose from for rotation
    :return: rotated image
    """
    for angle in angles:
        if not angle in {90, 180, 270}:
            raise ValueError("angles can only contain the values 90, 180, and 270.")
    p = np.random.rand(1)[0]
    if p < prob:
        # rotate the image
        number_of_rotations = np.random.choice((np.array(angles) / 90).astype(np.int64))
        return np.rot90(image_array, k=number_of_rotations)
    else:
        return image_array

