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

import keras
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Model
from keras.applications.resnet50 import ResNet50

from kplus.kclassifier.classifiers.AbstractClassifier import AbstractClassifier


class SimpleClassifier(AbstractClassifier):
    def __init__(self):
        AbstractClassifier.__init__(self)

    def _setup_model(self, parameters, is_training):
        input_layer = Input(shape=(224, 224, 3))
        base_model = ResNet50(
            weights=None, include_top=False, input_tensor=input_layer)

        x = base_model.output
        x = Flatten()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(5, activation='softmax')(x)

        self._keras_model = Model(input=base_model.input, output=predictions)
        self._keras_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['categorical_accuracy'])

        return (True)

    def evaluate(self, parameters):
        return (True)

    def predict(self, input_image):
        return (True)
