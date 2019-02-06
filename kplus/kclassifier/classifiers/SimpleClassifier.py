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

import keras
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import BatchNormalization
from keras import regularizers
from keras.models import Model

from kplus.kclassifier.classifiers.AbstractClassifier import AbstractClassifier


class SimpleClassifier(AbstractClassifier):
    def __init__(self):
        AbstractClassifier.__init__(self)

    def _setup_model(self, parameters, is_training):

        feature_extractor = parameters['model']['feature_extractor']
        if (not (feature_extractor in ['resnet_50'])):
            return (False)
        self.use_feature_extractor(feature_extractor)

        self._image_width = parameters['model']['image_width']
        self._image_height = parameters['model']['image_height']
        self._number_of_channels = parameters['model']['number_of_channels']

        input_shape = (self._image_height, self._image_width,
                       self._number_of_channels)
        input_layer = Input(
            name='input_layer', shape=input_shape, dtype='float32')

        self._feature_extractor.build(input_shape=input_shape)
        features = self._feature_extractor.extract_features(input_layer)

        x = BatchNormalization()(features)
        x = Flatten()(x)
        x = Dense(
            512,
            activation='relu',
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(0.00004))(x)
        x = BatchNormalization()(x)
        predictions = Dense(
            self._train_dataset.number_of_classes(),
            activation='softmax',
            kernel_initializer='he_normal')(x)

        self._keras_model = Model(input=input_layer, output=predictions)
        self._keras_model.summary()

        return (True)

    def predict(self, input_image):
        return (True)
