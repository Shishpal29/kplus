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

from keras.applications.resnet50 import ResNet50
from keras.models import Model

from kplus.kclassifier.models.AbstractModel import AbstractModel


class ResNet50Model(AbstractModel):

    __name = 'resnet_50'

    @classmethod
    def name(cls):
        return (ResNet50Model.__name)

    def __init__(self):
        AbstractModel.__init__(self)

    def build(self, input_shape):
        resnet50 = ResNet50(
            input_shape=input_shape, weights=None, include_top=False)

        # Remove the average pooling layer.
        resnet50.layers.pop()
        self.feature_extractor = Model(resnet50.layers[0].input,
                                       resnet50.layers[-1].output)
