import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def shuffle_arrays(arrays):
    if not arrays:
        return arrays

    size = arrays[0].shape[0]
    permutation = np.arange(size)
    np.random.shuffle(permutation)
    for i, array in enumerate(arrays):
        arrays[i] = array[permutation]

    return arrays


def shrink_arrays(arrays, shrink_size, is_shuffle=True):
    # if shrink_size in [0.0, 1.0] it specifies fraction of the array size to be extracted, if shrink_size is an
    # integer it specifies the size of the shrunk arrays

    if not arrays:
        return arrays

    if type(shrink_size)==float or type(shrink_size)==np.float64:
        assert(0.0 <= shrink_size <= 1.0)
        size = arrays[0].shape[0]
        shrunk_array_size = int(round(shrink_size * size))
    else:
        shrunk_array_size = shrink_size

    if is_shuffle:
        shuffled_arrays = shuffle_arrays(arrays)
    else:
        shuffled_arrays = arrays

    shrunk_arrays = []
    for array in shuffled_arrays:
        shrunk_arrays.append(array[:shrunk_array_size])

    return shrunk_arrays, shuffled_arrays


def convert_one_hot_to_label(one_hot_labels):
    return np.argwhere(one_hot_labels == 1)[:, 1]


def convert_label_to_one_hot(labels, n_classes):
    labels_arr = np.atleast_1d(labels)  # in case labels is a scalar
    one_hot_labels = np.zeros((labels_arr.shape[0], n_classes), dtype = np.int)
    one_hot_labels[np.arange(labels_arr.shape[0]), labels_arr] = 1
    return one_hot_labels


def next_batch_indices(sample_size, batch_size):
    if batch_size < sample_size:
        return np.random.choice(np.arange(sample_size), batch_size, replace=False)
    else:
        return np.random.permutation(sample_size)


def next_batch_indices_cycled(sample_size, batch_size, start_index):
    array = np.arange(sample_size)
    start = start_index
    stop = start + batch_size
    diff = stop - sample_size
    if diff <= 0:
        batch = array[start:stop]
        start_index += batch_size
    else:
        batch = np.concatenate((array[start:], array[:diff]))
        start_index = diff

    return batch, start_index

