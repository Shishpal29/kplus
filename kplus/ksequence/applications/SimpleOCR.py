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

from keras import backend as K
K.set_learning_phase(0)

from keras.optimizers import Adadelta

from kplus.ksequence.datasets.SimpleGenerator import SimpleGenerator
from kplus.ksequence.models.ModelFactory import ModelFactory

from kplus.core.AbstractApplication import AbstractApplication


class SimpleOCR(AbstractApplication):
    def __init__(self):
        self._image_width = 128
        self._image_height = 64
        self._downsample_factor = 4
        self._maximum_text_length = 9

    def _setup_model(self, parameters, is_training):
        model_name = parameters['model']['model_name']
        if (not (model_name in ['base', 'bidirectional', 'attention'])):
            return (False)

        feature_extractor = parameters['model']['feature_extractor']
        if (not (feature_extractor in ['simple_vgg', 'resnet_50'])):
            return (False)

        sequence_model = ModelFactory.simple_model(model_name)
        sequence_model.use_feature_extractor(feature_extractor)

        self._maximum_text_length = sequence_model.maximum_text_length()
        self._image_width = parameters['model']['image_width']
        self._image_height = parameters['model']['image_height']
        self._downsample_factor = parameters['model']['downsample_factor']

        input_shape = (self._image_width, self._image_height, 1)
        number_of_classes = SimpleGenerator.number_of_letters() + 1
        self._keras_model = sequence_model.keras_model(
            input_shape, number_of_classes, is_training)

        try:
            self._keras_model.load_weights(parameters['test']['model_name'])
        except:
            pass

        optimizer = Adadelta()

        # Dummy lambda function for the loss
        self._keras_model.compile(
            loss={
                'ctc': lambda y_true, y_pred: y_pred
            },
            optimizer=optimizer,
            metrics=['accuracy'])

        return (True)

    def _setup_train_dataset(self, train_dataset_dir, train_batch_size):
        self._train_dataset_generator = SimpleGenerator(
            train_dataset_dir, self._image_width, self._image_height,
            train_batch_size, self._downsample_factor,
            self._maximum_text_length)
        self._train_dataset_generator.build_data()
        return (True)

    def _setup_test_dataset(self, test_dataset_dir, test_batch_size):
        self._test_dataset_generator = SimpleGenerator(
            test_dataset_dir, self._image_width, self._image_height,
            test_batch_size, self._downsample_factor,
            self._maximum_text_length)
        self._test_dataset_generator.build_data()
        return (True)

    def _train_model(self, parameters):
        epoch = parameters['train']['max_number_of_epoch']
        train_batch_size = parameters['train']['batch_size']
        test_batch_size = parameters['test']['batch_size']
        self._keras_model.fit_generator(
            generator=self._train_dataset_generator.next_batch(),
            steps_per_epoch=int(
                self._train_dataset_generator.n / train_batch_size),
            callbacks=[
                self._checkpoint, self._early_stop, self._change_learning_rate,
                self._tensorboard
            ],
            epochs=epoch,
            validation_data=self._test_dataset_generator.next_batch(),
            validation_steps=int(
                self._test_dataset_generator.n / test_batch_size))

        return (True)

    def evaluate(self, parameters):
        pass

    def predict(self, input_image):
        pass
