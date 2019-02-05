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

import random
import numpy as np
import cv2

from kplus.datasets.AbstractBatchGenerator import AbstractBatchGenerator


class ImageBatchGenerator(AbstractBatchGenerator):
    def __init__(self):
        AbstractBatchGenerator.__init__(self)

        self._patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        self._image_width = 0
        self._image_height = 0
        self._number_of_channels = 0

    def _image_to_array(self, opencv_image, data_format=None):
        if (data_format is None):
            data_format = K.image_data_format()

        if (data_format not in ['channels_first', 'channels_last']):
            raise ValueError('Unknown data_format - ', data_format)

        # Numpy array x has format
        # (height, width, channel) - 'channels_last'
        # or
        # (channel, height, width) - 'channels_first'

        # OpenCV image has format (width, height, channel) and
        # image with channel=3 is in BGR format.
        if (len(opencv_image.shape) == 3):
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

        numpy_array = np.asarray(opencv_image, dtype=K.floatx())

        if (len(numpy_array.shape) == 3):
            if (data_format == 'channels_first'):
                numpy_array = numpy_array.transpose(2, 0, 1)
        elif (len(numpy_array.shape) == 2):
            if (data_format == 'channels_first'):
                numpy_array = numpy_array.reshape((1, numpy_array.shape[0],
                                                   numpy_array.shape[1]))
            else:
                numpy_array = numpy_array.reshape((numpy_array.shape[0],
                                                   numpy_array.shape[1], 1))
        else:
            raise ValueError('Unsupported image shape - ', numpy_array.shape)

        return (numpy_array)

    def _normalize(self, input_image):
        input_image = (input_image / 255.0) * 2.0 - 1.0
        return (input_image)

    def _augment_image(self, input_image, threshold=50.0):
        return (input_image)

    def _augment(self, input_image, threshold=50.0):
        augmented_image = input_image
        if (self.use_augmentation() and (random.randint(0, 100) > threshold)):
            augmented_image = self._augment_image(augmented_image, threshold)

        augmented_image = cv2.resize(augmented_image,
                                     (self._image_width, self._image_height))
        return (augmented_image)
