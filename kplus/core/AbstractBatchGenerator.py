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
import fnmatch
import numpy as np
import keras
import cv2


class AbstractBatchGenerator(keras.utils.Sequence):
    """
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):

        self.labels = labels
        self.list_IDs = list_IDs
        self.on_epoch_end()
    """

    _minimum_images = 1

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
        self._labels_filename = 'labels.txt'

        self._labels_to_class_names = None
        self._class_names_to_labels = None
        self._dataset = []
        self._identifiers = []
        self._number_of_classes = 0

        self._patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        self._image_width = 0
        self._image_height = 0
        self._number_of_channels = 0

        self._batch_size = 0
        self._shuffle = True

        self._random_seed = 7

    def labels_filename(self):
        return (self._labels_filename)

    def number_of_samples(self):
        return (len(self._dataset))

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

        self._number_of_classes = len(self._class_names_to_labels)
        #print(self._number_of_classes, self._labels_to_class_names, self._class_names_to_labels)

        return (True)

    def on_epoch_end(self):
        if (self._shuffle == True):
            np.random.shuffle(self._identifiers)
        #print(self._identifiers)

    def _load_dataset(self, dataset_dir):
        if ((not os.path.exists(dataset_dir))
                or (not os.path.isdir(dataset_dir))):
            return (False)

        self._dataset = []
        self._identifiers = []
        class_names = [
            class_name for class_name in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, class_name))
        ]

        number_of_samples = 0
        for class_name in class_names:
            class_source_dir = os.path.join(dataset_dir, class_name)

            if ((not os.path.isdir(class_source_dir))
                    or (class_name not in self._class_names_to_labels)):
                continue

            number_of_images = 0
            class_images = []
            for pattern in self._patterns:
                images = fnmatch.filter(os.listdir(class_source_dir), pattern)
                current_images = len(images)
                if (current_images > 0):
                    class_images = class_images + images
                number_of_images = number_of_images + current_images

            if (number_of_images >= AbstractBatchGenerator._minimum_images):
                for image in class_images:
                    source_file_name = os.path.join(class_source_dir, image)
                    class_label = self._class_names_to_labels[class_name]
                    current_data = {
                        'filename': source_file_name,
                        'label': class_label
                    }
                    #print(source_file_name, class_label)
                    self._dataset.append(current_data)
                    number_of_samples = number_of_samples + 1
                    self._identifiers.append(number_of_samples)

        #print(self._dataset)
        #print(self._identifiers)

        self.on_epoch_end()

        return (True)

    def _load_split(self, split_name, parameters):
        dataset_dir = parameters[split_name]['dataset_dir']
        train_dataset_dir = parameters[AbstractBatchGenerator.
                                       train_split_name()]['dataset_dir']
        if (not self._read_labels(train_dataset_dir)):
            return (False)

        self._image_width = parameters['model']['image_width']
        self._image_height = parameters['model']['image_height']
        self._number_of_channels = parameters['model']['number_of_channels']

        self._batch_size = parameters[split_name]['batch_size']

        return (self._load_dataset(dataset_dir))

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
        return (int(np.ceil(self.number_of_samples() / self._batch_size)))

    def __getitem__(self, index):
        lower_bound = index * self._batch_size
        upper_bound = (index + 1) * self._batch_size

        if (upper_bound > self.number_of_samples()):
            upper_bound = self.number_of_samples()
            lower_bound = upper_bound - self._batch_size

        for train_instance in self.images[l_bound:r_bound]:
            pass

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

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
