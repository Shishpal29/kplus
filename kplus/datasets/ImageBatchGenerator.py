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

try:
    from PIL import ImageEnhance
    from PIL import Image as pil_image
except ImportError:
    pil_image = None
    ImageEnhance = None

from kplus.datasets.AbstractBatchGenerator import AbstractBatchGenerator


class ImageBatchGenerator(AbstractBatchGenerator):
    def __init__(self):
        AbstractBatchGenerator.__init__(self)

        self._patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        self._image_width = 0
        self._image_height = 0
        self._number_of_channels = 0

    def _load_image(self, image_path, color_mode='rgb'):

        if (pil_image is None):
            raise ImportError(
                'Could not import PIL.Image. The use of `_load_image` requires PIL.'
            )

        image = pil_image.open(image_path)
        if (color_mode == 'grayscale'):
            if (image.mode != 'L'):
                image = image.convert('L')
        elif (color_mode == 'rgba'):
            if (image.mode != 'RGBA'):
                image = image.convert('RGBA')
        elif (color_mode == 'rgb'):
            if (image.mode != 'RGB'):
                image = image.convert('RGB')
        else:
            raise ValueError(
                'color_mode must be "grayscale", "rgb", or "rgba".')

        width_height_tuple = (self._image_width, self._image_height)
        if (image.size != width_height_tuple):
            image = image.resize(width_height_tuple, pil_image.BILINEAR)
        return (image)

    def _image_to_array(self, image, data_format=None, data_type='float32'):
        if (data_format is None):
            data_format = K.image_data_format()

        if (data_format not in ['channels_first', 'channels_last']):
            raise ValueError('Unknown data_format - ', data_format)

        # Numpy array x has format
        # (height, width, channel) - 'channels_last'
        # or
        # (channel, height, width) - 'channels_first'
        # but original PIL image has format (width, height, channel)

        numpy_array = np.asarray(image, dtype=data_type)

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

    def _array_to_image(numpy_array,
                        data_format='channels_last',
                        scale=True,
                        dtype='float32'):
        if pil_image is None:
            raise ImportError(
                'Could not import PIL.Image. The use of `_array_to_image` requires PIL.'
            )

        numpy_array = np.asarray(numpy_array, dtype=dtype)
        if (numpy_array.ndim != 3):
            raise ValueError(
                'Expected image array to have rank 3 (single image). Got array with shape: %s'
                % (numpy_array.shape))

        if (data_format not in {'channels_first', 'channels_last'}):
            raise ValueError('Invalid data_format - %s' % data_format)

        # Numpy array x has format
        # (height, width, channel) - 'channels_last'
        # or
        # (channel, height, width) - 'channels_first'
        # but original PIL image has format (width, height, channel)
        if data_format == 'channels_first':
            numpy_array = numpy_array.transpose(1, 2, 0)

        if scale:
            numpy_array = numpy_array + max(-np.min(numpy_array), 0)
            x_max = np.max(numpy_array)
            if x_max != 0:
                numpy_array /= x_max
            numpy_array *= 255

        if (numpy_array.shape[2] == 4):
            # RGBA
            return pil_image.fromarray(numpy_array.astype('uint8'), 'RGBA')
        elif (numpy_array.shape[2] == 3):
            # RGB
            return pil_image.fromarray(numpy_array.astype('uint8'), 'RGB')
        elif (numpy_array.shape[2] == 1):
            # grayscale
            return pil_image.fromarray(numpy_array[:, :, 0].astype('uint8'),
                                       'L')
        else:
            raise ValueError(
                'Unsupported channel number - %s' % (numpy_array.shape[2]))

    def _normalize(self, input_image):
        input_image = (input_image / 255.0) * 2.0 - 1.0
        return (input_image)

    def _augment_image(self, input_image):
        return (input_image)

    def _augment(self, input_image):
        augmented_image = input_image
        if (self.use_augmentation()):
            augmented_image = self._augment_image(augmented_image)
        return (augmented_image)
