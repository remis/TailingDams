import itertools

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import sys
sys.path.extend(['../../TailingDamDetection/'])

from dam_detection_sliding_window.postprocessor import PostProcessor
from utils.ml_functions import softmax

np.random.seed(1)
tf.set_random_seed(2)


def apply_for_sliding_window(model, image, patch_size, stride, batch_size, normalise_pathces=False, image_mean=None,
                             image_std=None):
    boxes, probs = [], []
    for patch_coords_batch, patch_batch in tqdm(sequential_pass_generator(
            image=image, patch_size=patch_size, stride=stride, batch_size=batch_size, normalise=normalise_pathces,
            image_mean=image_mean, image_std=image_std)):
        boxes.extend(patch_coords_batch)
        probs.extend(softmax(model.predict(patch_batch), axis=1))
    boxes = np.stack(boxes, axis=0)
    probs = np.stack(probs, axis=0)
    return boxes, probs


def sequential_pass_generator(image, patch_size, stride, batch_size, normalise=False, image_mean=None, image_std=None):
    """note the different order of indexes in coords and patch ind, this was due to this input in tf non_max_suppression"""
    #todo consider returning views to reduce required memory. Check sklearn.feature_extraction.image.extract_patches
    batch_ind = 0
    patch_coords_batch = []
    patch_batch = []
    img_width, img_height, _ = image.shape
    for top_left_border_x, top_left_border_y in itertools.product(
            range(0, img_width - patch_size[0], stride),
            range(0, img_height - patch_size[1], stride)):

        patch_coords_batch.append(np.array([top_left_border_x, top_left_border_y, top_left_border_x + patch_size[0],
                                            top_left_border_y + patch_size[1]]))
        patch_batch.append(image[top_left_border_x:top_left_border_x + patch_size[0],
                                 top_left_border_y:top_left_border_y + patch_size[1], :])
        batch_ind = batch_ind + 1
        if batch_ind >= batch_size:
            patch_coords_batch_np = np.stack(patch_coords_batch, axis=0)
            patch_batch_np = np.stack(patch_batch, axis=0)
            if normalise:
                patch_batch_np -= image_mean
                patch_batch_np /= image_std
            yield patch_coords_batch_np, patch_batch_np
            batch_ind = 0
            patch_coords_batch = []
            patch_batch = []

    #yield last patch
    if len(patch_coords_batch) > 0:
        patch_coords_batch_np = np.stack(patch_coords_batch, axis=0)
        patch_batch_np = np.stack(patch_batch, axis=0)
        if normalise:
            patch_batch_np -= image_mean
            patch_batch_np /= image_std
    yield patch_coords_batch_np, patch_batch_np


# use Pipeline instead
def wrap_predictions(image_shape, boxes, probs, output_mode='heatmaps', heatmap_mode='max', max_output_size=250,
                     class_label=0):

    postprocessor = PostProcessor(boxes=boxes, probs=probs[:, class_label],
                                  original_2dimage_shape=image_shape)
    if output_mode == 'heatmaps':
        predictions = postprocessor.heatmaps(mode=heatmap_mode)
    elif output_mode == 'non_max_suppression':
        predictions = postprocessor.non_max_suppression(max_output_size=max_output_size)
    else:
        # not implemented
        predictions = []

    return predictions
