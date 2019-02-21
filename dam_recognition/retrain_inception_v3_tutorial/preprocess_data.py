import gdal
import numpy as np
import config
import os
from PIL import Image

path_upwards = '../../'
path_to_images = path_upwards + config.data_path + config.image_path
path_to_output_images = path_upwards + config.data_path + config.preprocessed_image_path

image_value_min = 0
image_value_max = 7000

for subdir in next(os.walk(path_to_images))[1]:
    if not os.path.exists(path_to_output_images + subdir):
        os.makedirs(path_to_output_images + subdir)

    for file in os.listdir(path_to_images + subdir):
        img_array = np.transpose(np.array(gdal.Open(path_to_images + subdir + '/' + file).ReadAsArray()), axes=(1, 2, 0))

        # normalise image
        img_array = (img_array - image_value_min) / (image_value_max - image_value_min)

        # convertion to [0, 255]
        img_array = (img_array * 255).astype(np.uint8)

        img = Image.fromarray(img_array)
        img.save(path_to_output_images + subdir + '/' + file[:-4] + '.jpg')



