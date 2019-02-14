# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K

import numpy as np
import cv2

from kplus.datasets.ImageBatchGenerator import ImageBatchGenerator


class AugmentedBatchGenerator(ImageBatchGenerator):
    def __init__(self):
        ImageBatchGenerator.__init__(self)

    def _can_augment(self):
        return (np.random.uniform(0.0, 1.0) < 0.5)

    def _random_brightness(self, input_image, maximum_delta=32.0):
        if ((not self._can_augment()) or (maximum_delta < 0)):
            return (input_image)

        data_type = input_image.dtype
        input_image = input_image.astype('float32')

        value = np.random.uniform(-1.0 * maximum_delta, +1.0 * maximum_delta)
        input_image = input_image + value

        if (data_type == np.uint8):
            input_image = np.clip(input_image, 0, 255)

        return (input_image.astype(data_type))

    def _random_saturation(self, input_image, alpha=0.4):

        if ((not self._can_augment()) or (alpha >= 1.0)):
            return (input_image)

        value = 1.0 + np.random.uniform(-1.0 * alpha, +1.0 * alpha)

        data_type = input_image.dtype
        gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

        output_image = input_image * value + (gray_image *
                                              (1.0 - value))[:, :, np.newaxis]

        return (output_image.astype(data_type))

    def _random_hue(self, input_image, maximum_delta=0.2):

        if (not self._can_augment()):
            return (input_image)

        hue_value = np.random.uniform(-1.0 * maximum_delta,
                                      +1.0 * maximum_delta)

        hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)

        # OpenCV uses 0-179 degree instead of 0-359 degree
        hsv_image[..., 0] = (hsv_image[..., 0] + hue_value) % 180

        output_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

        return (output_image)

    def _random_contrast(self, input_image, lower_limit=0.5, upper_limit=1.5):

        if ((not self._can_augment()) or (lower_limit > upper_limit)):
            return (input_image)

        value = np.random.uniform(lower_limit, upper_limit)
        data_type = input_image.dtype
        input_image = input_image.astype('float32')

        mean = np.mean(input_image, axis=(0, 1), keepdims=True)
        input_image = (input_image - mean) * value + mean

        if (data_type == np.uint8):
            input_image = np.clip(input_image, 0, 255)

        return (input_image.astype(data_type))

    def _distort_color(self, input_image):
        choice = np.random.choice([0, 1, 2, 3])
        if (choice == 0):
            input_image = self._random_brightness(input_image)
            input_image = self._random_saturation(input_image)
            input_image = self._random_hue(input_image)
            input_image = self._random_contrast(input_image)
        elif (choice == 1):
            input_image = self._random_saturation(input_image)
            input_image = self._random_brightness(input_image)
            input_image = self._random_contrast(input_image)
            input_image = self._random_hue(input_image)
        elif (choice == 2):
            input_image = self._random_contrast(input_image)
            input_image = self._random_hue(input_image)
            input_image = self._random_brightness(input_image)
            input_image = self._random_saturation(input_image)
        else:
            input_image = self._random_hue(input_image)
            input_image = self._random_saturation(input_image)
            input_image = self._random_contrast(input_image)
            input_image = self._random_brightness(input_image)

        return (input_image)

    def _random_flip_left_right(self, input_image):
        if (self._can_augment()):
            input_image = cv2.flip(input_image, 1)
        return (input_image)

    def _random_resize(self, input_image):
        image_height, image_width = input_image.shape[:2]
        image_area = image_width * image_height
        maximum_attempts = 10
        output_image = input_image
        for attempt in range(maximum_attempts):

            target_area = np.random.uniform(0.5, 1.0) * image_area
            aspect_ratio = np.random.uniform(3.0 / 4.0, 4.0 / 3.0)

            new_width = int(np.sqrt(target_area * aspect_ratio))
            new_height = int(np.sqrt(target_area / aspect_ratio))

            if (self._can_augment()):
                new_width, new_height = new_height, new_width

            x1 = y1 = 0
            if ((new_height <= image_height) and (new_width <= image_width)):
                if (image_width == new_width):
                    x1 = 0
                else:
                    np.random.randint(0, image_width - new_width)

                if (image_height == new_height):
                    y1 = 0
                else:
                    np.random.randint(0, image_height - new_height)

                output_image = input_image[y1:(y1 + new_height), x1:(
                    x1 + new_width)]
                break

        output_image = self._resize(output_image)
        return (output_image)

    def _augment_image(self, input_image):
        output_image = self._random_resize(input_image)
        output_image = self._random_flip_left_right(output_image)
        output_image = self._distort_color(output_image)
        return (output_image)