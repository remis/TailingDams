import os

import h5py

import config
import numpy as np
import gdal

from dam_detection_sliding_window.predict_on_sliding_window import wrap_predictions, apply_for_sliding_window
from data_io.gdal_io import GdalIO
from nn_architecture.aka_lenet import cnn_adjust_lr

path_upwards = '../../'
experiment_name = 'two_classes_cv_smaller_images'

path_to_train_data = path_upwards + config.data_path + config.experiment_data + experiment_name + '/'
path_to_train_results = path_upwards + config.result_path + config.experiment_data + experiment_name + '/'
path_to_test_images = path_upwards + config.data_path + config.test_image_path
path_to_results = path_upwards + config.result_path + config.experiment_data + experiment_name + '/'
path_to_result_heatmaps = path_to_results + 'heatmaps/'
path_to_result_non_max_suppressed = path_to_results + 'non_max_suppressed_detection/'
if not os.path.exists(path_to_result_heatmaps):
    os.makedirs(path_to_result_heatmaps)
if not os.path.exists(path_to_result_non_max_suppressed):
    os.makedirs(path_to_result_non_max_suppressed)

n_runs = 10

best_accuracy = 0
best_model_id = -1

# determine best model during training
for run in range(n_runs):
    cur_results = np.load(path_to_train_results + 'results_{}_run.npz'.format(run))

    test_accuracy = cur_results['test_accuracy'][-1]

    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_model_id = run

train_data = np.load(path_to_train_data + 'train_test_data_{}.npz'.format(best_model_id))
image_mean = train_data['image_mean']
image_std = train_data['image_std']
patch_width, patch_height, patch_channels = train_data['train_images'][0].shape

model = cnn_adjust_lr(n_classes=train_data['train_labels'].shape[1],
                      input_shape=(patch_width, patch_height, patch_channels),
                      lr=1e-4)
model.load_weights(path_to_train_results + 'trained_model/weights_{}_run.ckpt'.format(best_model_id))

writer = GdalIO()

for file in os.listdir(path_to_test_images):
    image = np.transpose(np.array(gdal.Open(path_to_test_images + file).ReadAsArray()),
                         axes=(1, 2, 0)).astype(np.float64)
    writer.parse_meta_with_gdal(path_to_test_images + file)

    pred_boxes, pred_probs = apply_for_sliding_window(model, image, patch_size=(patch_width, patch_height),
                                                      stride=patch_width, batch_size=8, normalise_pathces=True,
                                                      image_mean=image_mean, image_std=image_std)

    with h5py.File(path_to_results + 'predicted_boxes.h5', 'w') as hf:
        hf.create_dataset("dataset", data=pred_boxes)

    with h5py.File(path_to_results + 'predicted_probs.h5', 'w') as hf:
        hf.create_dataset("dataset", data=pred_probs)

    image_width, image_height, _ = image.shape
    heatmap_predictions = wrap_predictions((image_width, image_height), pred_boxes, pred_probs,
                                           output_mode='heatmaps', heatmap_mode='mean',
                                           class_label=1)

    writer.write_surface(path_to_result_heatmaps + 'heatmap_' + file, heatmap_predictions)

    non_max_suppressed_predictions = wrap_predictions((image_width, image_height), pred_boxes, pred_probs,
                                                      output_mode='non_max_suppression', max_output_size=250,
                                                      class_label=1)

    writer.write_surface(path_to_result_non_max_suppressed + 'detection_' + file, non_max_suppressed_predictions)




