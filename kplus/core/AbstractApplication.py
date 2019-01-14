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

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard


class AbstractApplication(object):
    def __init__(self):
        self._checkpoint = None
        self._early_stop = None
        self._tensorboard = None

        self._train_dataset_generator = None
        self._test_dataset_generator = None

    def _setup_early_stop(self):
        self._early_stop = EarlyStopping(
            monitor='loss', min_delta=0.001, patience=4, mode='min', verbose=1)
        return (True)

    def _setup_model_checkpoint(self, checkpoint_path):
        self._checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='loss',
            verbose=1,
            mode='min',
            period=1)
        return (True)

    def _setup_tensorboard(self, tensorboard_path):
        self._tensorboard = TensorBoard(
            log_dir=tensorboard_path,
            histogram_freq=0,
            write_graph=True,
            write_images=False)
        return (True)

    def _change_learning_rate(self):
        return (True)

    def _setup_train_dataset(self, train_dataset_dir):
        return (False)

    def _setup_test_dataset(self, test_dataset_dir):
        return (False)

    def train(self, parameters):
        raise NotImplementedError('Must be implemented by the subclass.')

    def evaluate(self, parameters):
        raise NotImplementedError('Must be implemented by the subclass.')

    def predict(self, input_image):
        raise NotImplementedError('Must be implemented by the subclass.')
