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
from kplus.kclassifier.models.ModelFactory import ModelFactory


class AbstractClassifier(AbstractApplication):
    def __init__(self):
        AbstractApplication.__init__(self)
        self._image_width = 224
        self._image_height = 224
        self._number_of_channels = 3

        self._feature_extractor = None

    def use_feature_extractor(self, feature_extractor):
        self._feature_extractor = ModelFactory.simple_model(feature_extractor)

    def _batch_generator(self):
        return (ClassifierBatchGenerator())

    def _setup_loss_function(self, parameters):
        self._keras_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['categorical_accuracy'])
        return (True)
