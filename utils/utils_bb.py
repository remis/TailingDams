import numpy as np


def crop_array(input_array, centre, size):
    """
    return an array that is subarray of input_array with centre is in centre and of size equal size along the first 2
    dimensions, the rest are kept, e.g. a = np.reshape(np.arange(12), (3, 4)), then crop(a, (1, 1), (3, 3)) returns
    b = np.array([[0, 1, 2], [4, 5, 6], [8, 9, 10]])
    :param input_array: numpy nd-array with at least 2 dimensions
    :param centre: tuple (y, x) are coordinates of the centre of a new array in terms of the first 2 axis of input_array
    :param size: tuple (height, width) are shape in terms of the first 2 axis of a new array
    :return: numpy nd-array
    """
    height_start = centre[0] - size[0] // 2
    width_start = centre[1] - size[1] // 2
    if height_start < 0:
        height_start = 0
    if width_start < 0:
        width_start = 0

    height_finish = height_start + size[0]
    width_finish = width_start + size[1]
    if height_finish > input_array.shape[0]:
        height_finish = input_array.shape[0]
        height_start = height_finish - size[0]
    if width_finish > input_array.shape[1]:
        width_finish = input_array.shape[1]
        width_start = width_finish - size[0]

    cropped_array = np.copy(input_array[height_start:height_finish, width_start:width_finish, :])

    return cropped_array, np.arange(height_start, height_finish), np.arange(width_start, width_finish)

