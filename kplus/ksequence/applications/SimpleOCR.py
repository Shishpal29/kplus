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

from keras.optimizers import Adadelta

from kplus.ksequence.datasets.SimpleGenerator import SimpleGenerator
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
        optimizer = Adadelta()

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
        self._image_width = parameters['model']['image_width']
        self._image_height = parameters['model']['image_height']
        self._downsample_factor = parameters['model']['downsample_factor']

        input_shape = (self._image_width, self._image_height, 1)
        number_of_classes = len(self._model_letters) + 1

        self._keras_model = sequence_model.keras_model(
            input_shape, number_of_classes, self._maximum_text_length,
            is_training)

        return (True)

    def _setup_dataset(self, dataset_dir, batch_size):
        dataset = SimpleGenerator(self._model_letters, dataset_dir,
                                  self._image_width, self._image_height,
                                  batch_size, self._downsample_factor,
                                  self._maximum_text_length)
        dataset.build_data()
        return (dataset)

    def _setup_train_dataset(self, parameters):
        dataset_dir = parameters['train']['dataset_dir']
        batch_size = parameters['train']['batch_size']
        self._train_dataset = self._setup_dataset(dataset_dir, batch_size)
        return (True)

    def _setup_val_dataset(self, parameters):
        dataset_dir = parameters['val']['dataset_dir']
        batch_size = parameters['val']['batch_size']
        self._val_dataset = self._setup_dataset(dataset_dir, batch_size)
        return (True)

    def _setup_test_dataset(self, parameters):
        dataset_dir = parameters['test']['dataset_dir']
        batch_size = parameters['test']['batch_size']
        self._test_dataset = self._setup_dataset(dataset_dir, batch_size)
        return (True)

    def _train_model(self, parameters):
        epoch = parameters['train']['max_number_of_epoch']

        self._keras_model.fit_generator(
            generator=self._train_dataset.next_batch(),
            steps_per_epoch=self._train_dataset.steps_per_epoch(),
            callbacks=[
                self._checkpoint, self._early_stop, self._change_learning_rate,
                self._tensorboard
            ],
            epochs=epoch,
            validation_data=self._val_dataset.next_batch(),
            validation_steps=self._val_dataset.steps_per_epoch())

        return (True)

    def _evaluate_model(self, parameters):

        scores = self._keras_model.evaluate_generator(
            self._test_dataset.next_batch(),
            self._test_dataset.steps_per_epoch())
        print(scores)
        return (True)

    def predict(self, input_image):
        return (True)
