import numpy as np
import gdal
from PIL import Image
import os
from tempfile import NamedTemporaryFile

path_upwards = '../../'
import sys
sys.path.extend([path_upwards + '../TailingDamDetection/'])

import config

from utils.utils_dataset_processing import shuffle_arrays, convert_label_to_one_hot

rseed = 100
np.random.seed(rseed)


experiment_name = 'cv_10_folds'

path_to_images = path_upwards + config.data_path + config.train_image_path
path_to_output = path_upwards + config.data_path + config.experiment_data + experiment_name + '/'

if not os.path.exists(path_to_output):
    os.makedirs(path_to_output)

image_width = 236
image_height = 236
n_bands = 3
n_images = 890
n_test = 100
n_cv_runs = 10

images = np.zeros((n_images, image_width, image_height, n_bands), dtype=np.float64)
labels = -1 * np.ones((n_images,), dtype=np.int64)
char_labels = []

opt_string = '-outsize {} {}'.format(image_height, image_width)
image_counter = 0
for label_num, subdir in enumerate(next(os.walk(path_to_images))[1]):
    char_labels.append(subdir)

    for file in os.listdir(path_to_images + subdir):

        img_dataset = gdal.Translate(NamedTemporaryFile(delete=False).name,
                                     gdal.Open(path_to_images + subdir + '/' + file, gdal.GA_ReadOnly),
                                     options=opt_string)

        img_array = np.transpose(np.array(img_dataset.ReadAsArray()), axes=(1, 2, 0))

        images[image_counter, :, :, :] = img_array
        labels[image_counter] = label_num
        image_counter += 1

# write dictionary of labels
file = open(path_to_output + "label_dictionary.txt", "w")

for code, label in enumerate(char_labels):
    file.write('{}: {}\n'.format(code, label))

file.close()

labels = convert_label_to_one_hot(labels, len(char_labels))


for cv_run in range(n_cv_runs):
    shuffled_arrays = shuffle_arrays([images, labels])
    images = shuffled_arrays[0]
    labels = shuffled_arrays[1]

    test_images = images[-n_test:]
    test_labels = labels[-n_test:]

    train_images = images[:-n_test]
    train_labels = labels[:-n_test]

    # nomalisation
    image_mean = np.mean(train_images, axis=0)
    image_std = np.std(train_images, axis=0)

    normalised_train_images = np.copy(train_images)
    normalised_train_images -= image_mean
    normalised_train_images /= image_std

    normalised_test_images = np.copy(test_images)
    normalised_test_images -= image_mean
    normalised_test_images /= image_std

    np.savez(path_to_output + 'train_test_data_{}'.format(cv_run),
             train_images=normalised_train_images,
             train_labels=train_labels,
             test_images=normalised_test_images,
             test_labels=test_labels,
             image_mean=image_mean,
             image_std=image_std)







