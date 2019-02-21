import gdal
import numpy as np
import config
import os

path_upwards = '../'
path_to_images = path_upwards + config.data_path + config.image_path

n_bands = 3

image_value_band_min = 7000 * np.ones((n_bands,), dtype=np.int64)
image_value_band_max = np.zeros((n_bands,), dtype=np.int64)
image_value_band_mean = np.zeros((n_bands,), dtype=np.int64)


image_width_min = 250
image_width_max = 0
image_width_mean = 0

image_height_min = 250
image_height_max = 0
image_height_mean = 0


image_counter = 1

for subdir in next(os.walk(path_to_images))[1]:

    for file in os.listdir(path_to_images + subdir):
        img_array = np.transpose(np.array(gdal.Open(path_to_images + subdir + '/' + file).ReadAsArray()),
                                 axes=(1, 2, 0))
        cur_height, cur_width, _ = img_array.shape

        cur_value_band_min = np.zeros((n_bands,), dtype=np.int64)
        cur_value_band_max = np.zeros((n_bands,), dtype=np.int64)
        cur_value_band_mean = np.zeros((n_bands,), dtype=np.int64)
        for band in range(n_bands):
            cur_value_band_min[band] = np.min(img_array[:, :, band])
            cur_value_band_max[band] = np.max(img_array[:, :, band])
            cur_value_band_mean[band] = np.max(img_array[:, :, band])

        # update
        if cur_height > image_height_max:
            image_height_max = cur_height
        elif cur_height < image_height_min:
            image_height_min = cur_height

        image_height_mean = image_height_mean * (image_counter - 1) / image_counter + cur_height / image_counter

        if cur_width > image_width_max:
            image_width_max = cur_width
        elif cur_width < image_width_min:
            image_width_min = cur_width

        image_width_mean = image_width_mean * (image_counter - 1) / image_counter + cur_width / image_counter

        for band in range(n_bands):
            if cur_value_band_min[band] < image_value_band_min[band]:
                image_value_band_min[band] = cur_value_band_min[band]

            if cur_value_band_max[band] > image_value_band_max[band]:
                image_value_band_max[band] = cur_value_band_max[band]

            image_value_band_mean[band] = image_value_band_mean[band] * (image_counter - 1) / image_counter + \
                cur_value_band_mean[band] / image_counter

        image_counter += 1


print('image width:')
print('min: {}'.format(image_width_min))
print('max: {}'.format(image_width_max))
print('mean: {}'.format(image_width_mean))

print('image height:')
print('min: {}'.format(image_height_min))
print('max: {}'.format(image_height_max))
print('mean: {}'.format(image_height_mean))

for band in range(n_bands):
    print('image band {}:'.format(band))
    print('min: {}'.format(image_value_band_min[band]))
    print('max: {}'.format(image_value_band_max[band]))
    print('mean: {}'.format(image_value_band_mean[band]))


# min width 236, min height 236, number of images 890





