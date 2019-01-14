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
from kplus.ksequence.parameter import *

from kplus.core.AbstractApplication import AbstractApplication


class SimpleOCR(AbstractApplication):
    def __init__(self):
        pass

    def _setup_train_dataset(self, train_dataset_dir):
        self._train_dataset_generator = SimpleGenerator(
            train_dataset_dir, img_w, img_h, batch_size, downsample_factor)
        self._train_dataset_generator.build_data()
        return (True)

    def _setup_test_dataset(self, test_dataset_dir):
        test_batch_size = val_batch_size
        self._test_dataset_generator = SimpleGenerator(
            test_dataset_dir, img_w, img_h, test_batch_size, downsample_factor)
        self._test_dataset_generator.build_data()
        return (True)

    def train(self, parameters):

        model_name = parameters['model']['model_name']
        if (not (model_name in ['base', 'bidirectional', 'attention'])):
            return (False)

        feature_extractor = parameters['model']['feature_extractor']
        if (not (feature_extractor in ['simple_vgg', 'resnet_50'])):
            return (False)

        sequence_model = ModelFactory.simple_model(model_name)
        sequence_model.use_feature_extractor(feature_extractor)
        keras_model = sequence_model.keras_model(is_training=True)

        try:
            keras_model.load_weights('LSTM+BN4--26--0.011.hdf5')
            print("...Previous weight data...")
        except:
            print("...New weight data...")
            pass

        ada = Adadelta()

        # Dummy lambda function for the loss
        keras_model.compile(
            loss={
                'ctc': lambda y_true, y_pred: y_pred
            },
            optimizer=ada,
            metrics=['accuracy'])

        train_root_dir = os.path.expanduser(parameters['train_root_dir'])
        checkpoint_path = os.path.join(
            train_root_dir, 'model--{epoch:03d}--{val_loss:.5f}.hdf5')
        tensorboard_path = os.path.join(train_root_dir, 'tensorboard')

        status = True
        status = self._setup_early_stop() and self._setup_model_checkpoint(
            checkpoint_path) and self._setup_tensorboard(
                tensorboard_path) and self._change_learning_rate() and status

        train_dataset_dir = parameters['train_dataset_dir']
        test_dataset_dir = parameters['test_dataset_dir']
        status = self._setup_train_dataset(
            train_dataset_dir) and self._setup_test_dataset(
                test_dataset_dir) and status

        epoch = parameters['max_number_of_epoch']
        keras_model.fit_generator(
            generator=self._train_dataset_generator.next_batch(),
            steps_per_epoch=int(self._train_dataset_generator.n / batch_size),
            callbacks=[self._checkpoint, self._early_stop, self._tensorboard],
            epochs=epoch,
            validation_data=self._test_dataset_generator.next_batch(),
            validation_steps=int(
                self._test_dataset_generator.n / val_batch_size))

        return (status)

    def evaluate(self, parameters):
        pass

    def predict(self, input_image):
        pass
