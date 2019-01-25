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

import cv2
import os, random
import numpy as np


class SimpleGenerator:
    def __init__(self, letters, img_dirpath, img_w, img_h, batch_size,
                 downsample_factor, max_text_len):
        self._letters = letters
        self._image_height = img_h
        self._image_width = img_w
        self._batch_size = batch_size
        self._maximum_text_length = max_text_len
        self._downsample_factor = downsample_factor
        self._img_dirpath = img_dirpath  # image dir path
        self._img_dir = os.listdir(self._img_dirpath)  # images list
        self._number_of_samples = len(self._img_dir)  # number of images
        self._indexes = list(range(self._number_of_samples))
        self._current_index = 0
        self._images = np.zeros((self._number_of_samples, self._image_height,
                                 self._image_width))
        self._texts = []

    def steps_per_epoch(self):
        return (int(self._number_of_samples / self._batch_size))

    def labels_to_text(self, labels):
        return ''.join(list(map(lambda x: self._letters[int(x)], labels)))

    def text_to_labels(self, text):
        return list(map(lambda x: self._letters.index(x), text))

    def build_data(self):
        print(self._number_of_samples, " Image Loading start...")
        index = 0
        for img_file in self._img_dir:
            img = cv2.imread(self._img_dirpath + img_file,
                             cv2.IMREAD_GRAYSCALE)
            if (img is None):
                continue
            img = cv2.resize(img, (self._image_width, self._image_height))
            img = img.astype(np.float32)
            img = (img / 255.0) * 2.0 - 1.0

            self._images[index, :, :] = img
            self._texts.append(img_file[0:-4])
            index = index + 1
        print(len(self._texts) == self._number_of_samples)
        print(self._number_of_samples, index, " Image Loading finish...")

    def next_sample(self):
        self._current_index += 1
        if self._current_index >= self._number_of_samples:
            self._current_index = 0
            random.shuffle(self._indexes)
        return self._images[self._indexes[self._current_index]], self._texts[
            self._indexes[self._current_index]]

    def next_batch(self):
        while True:
            X_data = np.zeros(
                [self._batch_size, self._image_width, self._image_height,
                 1])  # (bs, 128, 64, 1)
            Y_data = np.zeros([self._batch_size,
                               self._maximum_text_length])  # (bs, 9)
            input_length = np.ones((self._batch_size, 1)) * (
                self._image_width // self._downsample_factor - 2)  # (bs, 1)
            label_length = np.zeros((self._batch_size, 1))  # (bs, 1)

            for i in range(self._batch_size):
                img, text = self.next_sample()
                img = img.T
                img = np.expand_dims(img, -1)
                X_data[i] = img
                chars = self.text_to_labels(text)
                length = label_length[i] = len(text)
                Y_data[i][0:length] = chars[0:length]

            inputs = {
                'input_image': X_data,  # (bs, 128, 64, 1)
                'input_labels': Y_data,  # (bs, 8)
                'input_length': input_length,  # (bs, 1) -> value = 30
                'label_length': label_length  # (bs, 1) -> value = 8
            }
            outputs = {'ctc': np.zeros([self._batch_size])}
            yield (inputs, outputs)
