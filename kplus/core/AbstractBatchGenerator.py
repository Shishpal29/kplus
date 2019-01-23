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
import numpy as np
import keras
import cv2


class AbstractBatchGenerator(keras.utils.Sequence):
    """
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):

        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
    """

    def __init__(self):
        self._labels_filename = 'labels.txt'
        self._labels_to_class_names = None
        self._class_names_labels_to = None

        self._image_width = 0
        self._image_height = 0
        self._number_of_channels = 0

        self._random_seed = 7

    def labels_filename(self):
        return (self._labels_filename)

    def _generate_labels(self, source_root_dir, target_root_dir):

        class_names = []
        for class_name in os.listdir(source_root_dir):
            class_path = os.path.join(source_root_dir, class_name)
            if (os.path.isdir(class_path)):
                class_names.append(class_name)

        random.seed(self._random_seed)
        random.shuffle(class_names)

        labels_to_class_names = dict(zip(range(len(class_names)), class_names))
        labels_file_path = os.path.join(target_root_dir,
                                        self.labels_filename())

        with open(labels_file_path, 'w') as file_stream:
            for label in labels_to_class_names:
                class_name = labels_to_class_names[label]
                file_stream.write('%d:%s\n' % (label, class_name))

        return (True)

    def generate_dataset(self, source_root_dir, target_root_dir):

        if (not os.path.exists(source_root_dir)):
            return (False)

        target_root_dir = os.path.expanduser(target_root_dir)
        if (not os.path.exists(target_root_dir)):
            os.makedirs(target_root_dir)

        status = True
        status = self._generate_labels(source_root_dir,
                                       target_root_dir) and status

        return (status)

    def _has_labels(self, source_root_dir):
        labels_file_path = os.path.join(source_root_dir,
                                        self.labels_filename())
        status = os.path.exists(labels_file_path) and os.path.isfile(
            labels_file_path)
        return (status)

    def _read_labels(self, source_root_dir):

        if (not self._has_labels(source_root_dir)):
            return (False)

        labels_file_path = os.path.join(source_root_dir,
                                        self.labels_filename())

        with open(labels_file_path, 'r') as labels_file:
            lines = labels_file.readlines()

        self._labels_to_class_names = {}
        self._class_names_to_labels = {}
        for line in lines:
            line = line.strip('\n')
            index = line.index(':')
            self._labels_to_class_names[int(line[:index])] = str(
                line[index + 1:])
            self._class_names_to_labels[str(line[index + 1:])] = int(
                line[:index])

        print(self._labels_to_class_names, self._class_names_to_labels)

        return (True)

    def load(self, split_name, parameters):
        train_dataset_dir = parameters[split_name]['dataset_dir']
        if (not self._read_labels(train_dataset_dir)):
            return (False)

        self._image_width = parameters['model']['image_width']
        self._image_height = parameters['model']['image_height']
        self._number_of_channels = parameters['model']['number_of_channels']

        self._batch_size = parameters[split_name]['batch_size']

        return (True)

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) *
                               self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i, ] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
