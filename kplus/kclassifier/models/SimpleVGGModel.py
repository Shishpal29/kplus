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

from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda, BatchNormalization
from keras.models import Model

from kplus.kclassifier.models.AbstractModel import AbstractModel


class SimpleVGGModel(AbstractModel):

    __name = 'simple_vgg'

    @classmethod
    def name(cls):
        return (SimpleVGGModel.__name)

    def __init__(self):
        AbstractModel.__init__(self)

    def build(self, input_shape):

        simple_vgg_input = Input(
            name='simple_vgg_input', shape=input_shape,
            dtype='float32')  # (None, 64, 128, 1)

        inner = Conv2D(
            64, (3, 3),
            padding='same',
            name='conv1',
            kernel_initializer='he_normal')(
                simple_vgg_input)  # (None, 64, 128, 64)

        inner = BatchNormalization()(inner)

        inner = Activation('relu')(inner)

        inner = MaxPooling2D(
            pool_size=(2, 2), name='max1')(inner)  # (None, 32, 64, 64)

        inner = Conv2D(
            128, (3, 3),
            padding='same',
            name='conv2',
            kernel_initializer='he_normal')(inner)  # (None, 32, 64, 128)

        inner = BatchNormalization()(inner)

        inner = Activation('relu')(inner)

        inner = MaxPooling2D(
            pool_size=(2, 2), name='max2')(inner)  # (None, 16, 32, 128)

        inner = Conv2D(
            256, (3, 3),
            padding='same',
            name='conv3',
            kernel_initializer='he_normal')(inner)  # (None, 16, 32, 256)

        inner = BatchNormalization()(inner)

        inner = Activation('relu')(inner)

        inner = Conv2D(
            256, (3, 3),
            padding='same',
            name='conv4',
            kernel_initializer='he_normal')(inner)  # (None, 16, 32, 256)

        inner = BatchNormalization()(inner)

        inner = Activation('relu')(inner)

        inner = MaxPooling2D(
            pool_size=(2, 1), name='max3')(inner)  # (None, 8, 32, 256)

        inner = Conv2D(
            512, (3, 3),
            padding='same',
            name='conv5',
            kernel_initializer='he_normal')(inner)  # (None, 8, 32, 512)

        inner = BatchNormalization()(inner)

        inner = Activation('relu')(inner)

        inner = Conv2D(
            512, (3, 3), padding='same',
            name='conv6')(inner)  # (None, 8, 32, 512)

        inner = BatchNormalization()(inner)

        inner = Activation('relu')(inner)

        inner = MaxPooling2D(
            pool_size=(2, 1), name='max4')(inner)  # (None, 4, 32, 512)

        inner = Conv2D(
            512, (2, 2),
            padding='same',
            kernel_initializer='he_normal',
            name='con7')(inner)  # (None, 4, 32, 512)

        inner = BatchNormalization()(inner)

        simple_vgg_output = Activation('relu')(inner)

        self.feature_extractor = Model(simple_vgg_input, simple_vgg_output)
