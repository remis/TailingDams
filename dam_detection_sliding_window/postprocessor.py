from typing import Tuple

import tensorflow as tf
import numpy as np
from tqdm import tqdm


class PostProcessor:
    def __init__(self, boxes: np.array, probs: np.array, original_2dimage_shape: Tuple[int, int]):
        """
        :param boxes: 2d array of shape n_boxes, 4
        :param probs: 1d array of shape n_boxes
        :param original_2dimage_shape:
        """
        self.boxes = boxes
        self.probs = probs
        self.original_2dimage_shape = original_2dimage_shape

    def non_max_suppression(self, max_output_size: int):
        iou_threshold = 0.5
        score_threshold = float('-inf')

        with tf.Session() as sess:
            selected_indices = tf.image.non_max_suppression(
                self.boxes,
                self.probs,
                max_output_size=max_output_size,
                iou_threshold=iou_threshold,
                score_threshold=score_threshold,
                name=None
            )

            boxes_ind = selected_indices.eval(session=sess)

            res_im = np.zeros(self.original_2dimage_shape, dtype=np.float)
            for box_ind in boxes_ind:
                c_box = self.boxes[box_ind]
                res_im[c_box[0]:c_box[2], c_box[1]: c_box[3]] = self.probs[box_ind]

        return res_im

    def heatmaps(self, mode: str):
        res_im = np.zeros(self.original_2dimage_shape,
                          dtype=np.float)
        if mode == "max":
            for (index, borders) in enumerate(tqdm(self.boxes)):
                top_left_x, top_left_y, bottom_right_x, bottom_right_y = borders
                res_im[top_left_x:bottom_right_x, top_left_y:bottom_right_y] = np.maximum(
                    res_im[top_left_x:bottom_right_x, top_left_y:bottom_right_y],
                    self.probs[index] * np.ones_like(res_im[top_left_x:bottom_right_x, top_left_y:bottom_right_y]))
        elif mode == "mean":
            # todo add padding of the same pixel values
            stride = self.boxes[1, 1] - self.boxes[0, 1]
            patch_width = self.boxes[0, 2] - self.boxes[0, 0] #todo this is for square patches only now

            number_of_times_pixel_in_patch = int((patch_width / stride) ** 2)
            for (index, borders) in enumerate(tqdm(self.boxes)):
                top_left_x, top_left_y, bottom_right_x, bottom_right_y = borders
                res_im[top_left_x:bottom_right_x, top_left_y:bottom_right_y] \
                    = res_im[top_left_x:bottom_right_x, top_left_y:bottom_right_y] \
                      + np.log(self.probs[index] * np.ones_like(res_im[top_left_x:bottom_right_x, top_left_y:bottom_right_y]))
            res_im = np.exp(res_im / number_of_times_pixel_in_patch)

        return res_im
