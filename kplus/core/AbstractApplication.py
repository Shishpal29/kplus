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
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau


class AbstractApplication(object):
    def __init__(self):
        self._keras_model = None

        self._checkpoint = None
        self._early_stop = None
        self._change_learning_rate = None
        self._tensorboard = None

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

    def load_weights(self, weight_path):
        self._keras_model.load_weights(weight_path)

    def _setup_model_checkpoint(self, checkpoint_path):
        self._checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='loss',
            verbose=1,
            mode='auto',
            period=1)
        return (True)

    def _setup_tensorboard(self, tensorboard_path):
        self._tensorboard = TensorBoard(
            log_dir=tensorboard_path,
            histogram_freq=0,
            write_graph=True,
            write_images=False)
        return (True)

    def _setup_early_stop(self):
        #self._early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=5, mode='min', verbose=1)
        self._early_stop = EarlyStopping(
            monitor='val_loss',
            min_delta=0.0,
            patience=5,
            mode='auto',
            verbose=1)
        return (True)

    def _setup_learning_rate(self):
        self._change_learning_rate = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=3,
            verbose=1,
            mode='auto',
            epsilon=0.0001,
            cooldown=1,
            min_lr=1e-7)
        return (True)

    def _setup_callbacks(self, parameters):
        train_root_dir = os.path.expanduser(
            parameters['train']['train_root_dir'])
        checkpoint_path = os.path.join(train_root_dir,
                                       parameters['train']['model_filename'])
        tensorboard_path = os.path.join(train_root_dir, 'tensorboard')

        status = True
        status = self._setup_early_stop() and status
        status = self._setup_model_checkpoint(checkpoint_path) and status
        status = self._setup_tensorboard(tensorboard_path) and status
        status = self._setup_learning_rate() and status
        return (status)

    def _train_model(self, parameters):
        epoch = parameters['train']['max_number_of_epoch']
        self._keras_model.fit_generator(
            generator=self._train_dataset,
            steps_per_epoch=self._train_dataset.steps_per_epoch(),
            callbacks=[
                self._checkpoint, self._early_stop, self._change_learning_rate,
                self._tensorboard
            ],
            epochs=epoch,
            validation_data=self._val_dataset,
            validation_steps=self._val_dataset.steps_per_epoch())

        return (True)

    def _batch_generator(self):
        raise NotImplementedError('Must be implemented by the subclass.')

    def _setup_train_dataset(self, parameters):
        self._train_dataset = self._batch_generator()
        self._train_dataset.load_train_dataset(parameters)
        return (True)

    def _setup_val_dataset(self, parameters):
        self._val_dataset = self._batch_generator()
        self._val_dataset.load_val_dataset(parameters)
        return (True)

    def _setup_test_dataset(self, parameters):
        self._test_dataset = self._batch_generator()
        self._test_dataset.load_test_dataset(parameters)
        return (True)

    def _setup_train_datasets(self, parameters):
        self._setup_train_dataset(parameters)
        self._setup_val_dataset(parameters)
        return (True)

    def _setup_loss_function(self, parameters):
        raise NotImplementedError('Must be implemented by the subclass.')

    def _setup_model_parameters(self, parameters, is_training):
        return (True)

    def _setup_model(self, parameters, is_training):
        raise NotImplementedError('Must be implemented by the subclass.')

    def _load_model(self, parameters):
        status = True
        try:
            self.load_weights(parameters['model']['model_filename'])
        except:
            pass
        return (status)

    def train(self, parameters):
        K.set_learning_phase(1)

        status = True

        status = self._setup_model_parameters(
            parameters, is_training=True) and status
        if (not status):
            return (False)

        status = self._setup_train_datasets(parameters) and status
        if (not status):
            return (False)

        status = self._setup_model(parameters, is_training=True) and status
        if (not status):
            return (False)

        status = self._load_model(parameters) and status
        if (not status):
            return (False)

        status = self._setup_loss_function(parameters) and status
        if (not status):
            return (False)

        status = self._setup_callbacks(parameters) and status
        if (not status):
            return (False)

        status = self._train_model(parameters) and status
        if (not status):
            return (False)

        return (status)

    def evaluate(self, parameters):
        K.set_learning_phase(0)

        status = True

        status = self._setup_model_parameters(
            parameters, is_training=False) and status
        if (not status):
            return (False)

        status = self._setup_test_dataset(parameters) and status
        if (not status):
            return (False)

        status = self._setup_model(parameters, is_training=True) and status
        if (not status):
            return (False)

        status = self._load_model(parameters) and status
        if (not status):
            return (False)

        status = self._setup_loss_function(parameters) and status
        if (not status):
            return (False)

        status = self._evaluate_model(parameters) and status
        if (not status):
            return (False)

        return (status)

    def _evaluate_model(self, parameters):

        loss, accuracy = self._keras_model.evaluate_generator(
            self._test_dataset, self._test_dataset.steps_per_epoch())
        print('Test loss - ', str(loss))
        print('Test accuracy -', str(accuracy))

        return (True)

    def predict(self, input_image):
        raise NotImplementedError('Must be implemented by the subclass.')
