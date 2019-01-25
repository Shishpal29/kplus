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

from kplus.core.AbstractApplication import AbstractApplication
from kplus.kclassifier.datasets.ClassifierBatchGenerator import ClassifierBatchGenerator


class AbstractClassifier(AbstractApplication):
    def __init__(self):
        AbstractApplication.__init__(self)

    def _setup_train_dataset(self, parameters):
        status = True
        self._train_dataset_generator = ClassifierBatchGenerator()
        status = self._train_dataset_generator.load_train_dataset(
            parameters) and status
        retur(status)

    def _setup_val_dataset(self, parameters):
        status = True
        self._val_dataset_generator = ClassifierBatchGenerator()
        status = self._val_dataset_generator.load_val_dataset(
            parameters) and status
        retur(status)

    def _setup_test_dataset(self, parameters):
        status = True
        self._test_dataset_generator = ClassifierBatchGenerator()
        status = self._test_dataset_generator.load_test_dataset(
            parameters) and status
        retur(status)
