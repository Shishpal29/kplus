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

from kplus.datasets.AbstractBatchGenerator import AbstractBatchGenerator
from kplus.datasets.ImageBatchGenerator import ImageBatchGenerator


class ClassifierBatchGenerator(ImageBatchGenerator):

    _minimum_images = 1

    @classmethod
    def filename_tag(cls):
        return ('filename')

    @classmethod
    def label_tag(cls):
        return ('label')

    def __init__(self):
        ImageBatchGenerator.__init__(self)

        self._labels_filename = 'labels.txt'

        self._labels_to_class_names = None
        self._class_names_to_labels = None

        self._number_of_classes = 0

    def labels_filename(self):
        return (self._labels_filename)

    def number_of_classes(self):
        return (self._number_of_classes)

    def _generate_labels(self, source_root_dir, target_root_dir):

        class_names = []
        for class_name in os.listdir(source_root_dir):
            class_path = os.path.join(source_root_dir, class_name)
            if (os.path.isdir(class_path)):
                class_names.append(class_name)

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

        return (True)

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

            if (number_of_images >= ClassifierBatchGenerator._minimum_images):
                for image in class_images:
                    source_file_name = os.path.join(class_source_dir, image)
                    class_label = self._class_names_to_labels[class_name]
                    current_data = {
                        ClassifierBatchGenerator.filename_tag():
                        source_file_name,
                        ClassifierBatchGenerator.label_tag(): class_label
                    }

                    self._dataset.append(current_data)
                    self._identifiers.append(number_of_samples)
                    number_of_samples = number_of_samples + 1

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

    def __getitem__(self, batch_index):

        lower_bound = batch_index * self._batch_size
        upper_bound = (batch_index + 1) * self._batch_size

        if (upper_bound > self.number_of_samples()):
            upper_bound = self.number_of_samples()
            lower_bound = upper_bound - self._batch_size

        X = np.empty((self._batch_size, self._image_height, self._image_width,
                      self._number_of_channels))
        y = np.empty((self._batch_size, self.number_of_classes()), dtype=int)

        for index in range(lower_bound, upper_bound):
            target_index = index - lower_bound
            source_identifier = self._identifiers[index]

            filename = self._dataset[source_identifier][
                ClassifierBatchGenerator.filename_tag()]
            label = self._dataset[source_identifier][ClassifierBatchGenerator.
                                                     label_tag()]

            input_image = cv2.imread(filename, cv2.IMREAD_COLOR)
            input_image = self._augment(input_image)
            input_image = self._image_to_array(input_image)
            input_image = self._normalize(input_image)

            X[target_index] = input_image
            y[target_index] = keras.utils.np_utils.to_categorical(
                label, self.number_of_classes())

        return (X, y)
