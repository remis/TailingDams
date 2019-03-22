import numpy as np
import gdal
from PIL import Image
import config
import os
from tempfile import NamedTemporaryFile

from utils.utils_bb import crop_array
from utils.utils_dataset_processing import shuffle_arrays, convert_label_to_one_hot

rseed = 100
np.random.seed(rseed)

path_upwards = '../../'
experiment_name = 'two_classes_cv_smaller_images'

path_to_original_images = path_upwards + config.data_path + config.train_image_path
path_to_updated_dam_images = path_upwards + config.data_path + config.train_updated_dam_image_path
path_to_output = path_upwards + config.data_path + config.experiment_data + experiment_name + '/'

if not os.path.exists(path_to_output):
    os.makedirs(path_to_output)

image_width = 95
image_height = 95
n_bands = 3
n_sample_from_no_dams = 2
# dams + lookalikes + double no dams
n_images = 855 + 51 + 300 * n_sample_from_no_dams
n_test = 300
n_cv_runs = 10

images = np.zeros((n_images, image_width, image_height, n_bands), dtype=np.float64)
labels = -1 * np.ones((n_images,), dtype=np.int64)
char_labels = []

opt_string = '-outsize {} {}'.format(image_height, image_width)
image_counter = 0
# read lookalikes and no dams images
for label_num, subdir in enumerate(next(os.walk(path_to_original_images))[1]):
    if subdir == 'dams':
        # dams are in separate folder
        continue

    char_labels.append(subdir)

    for file in os.listdir(path_to_original_images + subdir):

        img_dataset = gdal.Translate(NamedTemporaryFile(delete=False).name,
                                     gdal.Open(path_to_original_images + subdir + '/' + file, gdal.GA_ReadOnly))

        img_array = np.transpose(np.array(img_dataset.ReadAsArray()), axes=(1, 2, 0))

        if subdir == 'looksLikeDam':
            # crop image from the centre
            img_array, _, _ = crop_array(img_array, (img_array.shape[0] // 2, img_array.shape[1] // 2),
                                         (image_height, image_width))

            images[image_counter, :, :, :] = img_array
            image_counter += 1

        elif subdir == 'noDam':
            # sample two random image patches of given size
            available_heights = np.arange(img_array.shape[0])
            available_widths = np.arange(img_array.shape[0])
            for sample_num in range(n_sample_from_no_dams):
                sample_centre = -1 * np.ones((2,), dtype=np.int64)
                sample_centre[0] = np.random.choice(available_heights, 1, replace=False)
                sample_centre[1] = np.random.choice(available_widths, 1, replace=False)

                sampled_img, used_heights, used_widths = crop_array(img_array, sample_centre,
                                                                    (image_height, image_width))

                available_heights = np.setdiff1d(available_heights, used_heights)
                available_widths = np.setdiff1d(available_widths, used_widths)

                images[image_counter, :, :, :] = sampled_img
                image_counter += 1


# combine lookalike and no dams
labels[:image_counter] = 0

# read dam images
for file in os.listdir(path_to_updated_dam_images):
    img_dataset = gdal.Translate(NamedTemporaryFile(delete=False).name,
                                 gdal.Open(path_to_updated_dam_images + file, gdal.GA_ReadOnly),
                                 options=opt_string)

    img_array = np.transpose(np.array(img_dataset.ReadAsArray()), axes=(1, 2, 0))

    images[image_counter, :, :, :] = img_array
    labels[image_counter] = 1
    image_counter += 1

char_labels = ['not_a_dam', 'dam']

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

    # normalisation
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







