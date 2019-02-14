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

import os

from keras.optimizers import Adam

from kplus.ksequence.datasets.SequenceBatchGenerator import SequenceBatchGenerator
from kplus.ksequence.models.ModelFactory import ModelFactory

from kplus.core.AbstractApplication import AbstractApplication


class SimpleOCR(AbstractApplication):
    def __init__(self):
        AbstractApplication.__init__(self)
        self._image_width = 128
        self._image_height = 64
        self._downsample_factor = 4
        self._maximum_text_length = 9
        self._model_letters = None

    def _setup_loss_function(self, parameters):
        base_learning_rate = parameters['train']['base_learning_rate']
        optimizer = Adam(lr=base_learning_rate)

        # Dummy lambda function for the loss
        self._keras_model.compile(
            loss={
                'ctc': lambda y_true, y_pred: y_pred
            },
            optimizer=optimizer,
            metrics=['accuracy'])
        return (True)

    def _setup_model_parameters(self, parameters, is_training):
        model_letters = parameters['model']['letters']
        self._model_letters = [letter for letter in model_letters]
        return (True)

    def _setup_model(self, parameters, is_training):
        model_name = parameters['model']['model_name']
        if (not (model_name in ['base', 'bidirectional', 'attention'])):
            return (False)

        feature_extractor = parameters['model']['feature_extractor']
        if (not (feature_extractor in ['simple_vgg', 'resnet_50'])):
            return (False)

        sequence_model = ModelFactory.simple_model(model_name)
        sequence_model.use_feature_extractor(feature_extractor)

        self._maximum_text_length = parameters['model']['maximum_text_length']
        self._downsample_factor = parameters['model']['downsample_factor']

        self._image_width = parameters['model']['image_width']
        self._image_height = parameters['model']['image_height']
        self._number_of_channels = parameters['model']['number_of_channels']

        # (image height, image width, number of channels)
        # is converted to
        # (image width, image height, number of channels)
        # for better accuracy.
        #input_shape = (self._image_height, self._image_width, self._number_of_channels)
        input_shape = (self._image_width, self._image_height,
                       self._number_of_channels)

        number_of_classes = len(self._model_letters) + 1

        self._keras_model = sequence_model.keras_model(
            input_shape, number_of_classes, self._maximum_text_length,
            is_training)

        self._keras_model.summary()

        return (True)

    def _batch_generator(self):
        return (SequenceBatchGenerator())

    def predict(self, input_image):
        return (True)
