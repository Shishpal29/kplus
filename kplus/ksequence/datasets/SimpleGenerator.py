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

    __all_letters = "adefghjknqrstwABCDEFGHIJKLMNOPZ0123456789"
    __letters = [letter for letter in __all_letters]

    @classmethod
    def number_of_letters(cls):
        return (len(SimpleGenerator.__letters))

    @classmethod
    def labels_to_text(cls, labels):
        return ''.join(
            list(map(lambda x: SimpleGenerator.__letters[int(x)], labels)))

    @classmethod
    def text_to_labels(cls, text):
        return list(map(lambda x: SimpleGenerator.__letters.index(x), text))

    def __init__(self, img_dirpath, img_w, img_h, batch_size,
                 downsample_factor, max_text_len):
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.downsample_factor = downsample_factor
        self.img_dirpath = img_dirpath  # image dir path
        self.img_dir = os.listdir(self.img_dirpath)  # images list
        self.n = len(self.img_dir)  # number of images
        self.indexes = list(range(self.n))
        self.cur_index = 0
        self.imgs = np.zeros((self.n, self.img_h, self.img_w))
        self.texts = []

    def build_data(self):
        print(self.n, " Image Loading start...")
        for i, img_file in enumerate(self.img_dir):
            img = cv2.imread(self.img_dirpath + img_file, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (self.img_w, self.img_h))
            img = img.astype(np.float32)
            img = (img / 255.0) * 2.0 - 1.0

            self.imgs[i, :, :] = img
            self.texts.append(img_file[0:-4])
        print(len(self.texts) == self.n)
        print(self.n, " Image Loading finish...")

    def next_sample(self):
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)
        return self.imgs[self.indexes[self.cur_index]], self.texts[
            self.indexes[self.cur_index]]

    def next_batch(self):
        while True:
            X_data = np.ones([self.batch_size, self.img_w, self.img_h,
                              1])  # (bs, 128, 64, 1)
            Y_data = np.ones([self.batch_size, self.max_text_len])  # (bs, 9)
            input_length = np.ones((self.batch_size, 1)) * (
                self.img_w // self.downsample_factor - 2)  # (bs, 1)
            label_length = np.zeros((self.batch_size, 1))  # (bs, 1)

            for i in range(self.batch_size):
                img, text = self.next_sample()
                img = img.T
                img = np.expand_dims(img, -1)
                X_data[i] = img
                Y_data[i] = SimpleGenerator.text_to_labels(text)
                label_length[i] = len(text)

            inputs = {
                'input_image': X_data,  # (bs, 128, 64, 1)
                'input_labels': Y_data,  # (bs, 8)
                'input_length': input_length,  # (bs, 1) -> value = 30
                'label_length': label_length  # (bs, 1) -> value = 8
            }
            outputs = {'ctc': np.zeros([self.batch_size])}
            yield (inputs, outputs)
