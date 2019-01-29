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
import random
import math

import keras


class AbstractBatchGenerator(keras.utils.Sequence):
    @classmethod
    def train_split_name(cls):
        return ('train')

    @classmethod
    def val_split_name(cls):
        return ('val')

    @classmethod
    def test_split_name(cls):
        return ('test')

    def __init__(self):

        self._dataset = []
        self._identifiers = []

        self._batch_size = 0
        self._shuffle = True

        self._random_seed = 7

    def number_of_samples(self):
        return (len(self._identifiers))

    def batch_size(self):
        return (self._batch_size)

    def steps_per_epoch(self):
        return (int(self.number_of_samples() / self.batch_size()))

    def on_epoch_end(self):
        if (self._shuffle == True):
            random.shuffle(self._identifiers)

    def _load_dataset(self, dataset_dir):
        raise NotImplementedError('Must be implemented by the subclass.')

    def _load_split(self, split_name, parameters):
        raise NotImplementedError('Must be implemented by the subclass.')

    def load_train_dataset(self, parameters):
        split_name = AbstractBatchGenerator.train_split_name()
        return (self._load_split(split_name, parameters))

    def load_val_dataset(self, parameters):
        split_name = AbstractBatchGenerator.val_split_name()
        return (self._load_split(split_name, parameters))

    def load_test_dataset(self, parameters):
        split_name = AbstractBatchGenerator.test_split_name()
        return (self._load_split(split_name, parameters))

    def __len__(self):
        return (int(math.ceil(self.number_of_samples() / self._batch_size)))

    def __getitem__(self, batch_index):
        raise NotImplementedError('Must be implemented by the subclass.')
