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

from keras import backend as K
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda, BatchNormalization
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import LSTM

from kplus.ksequence.parameter import *

from kplus.kclassifier.models.ModelFactory import ModelFactory

K.set_learning_phase(0)


class BaseModel(object):

    __name = 'base'

    def __init__(self):
        self._feature_extractor = None
        self._input_shape = (img_w, img_h, 1)  # (128, 64, 1)

    @classmethod
    def name(cls):
        return (BaseModel.__name)

    def use_feature_extractor(self, feature_extractor):
        self._feature_extractor = ModelFactory.simple_model(
            feature_extractor, input_shape=self._input_shape)

    def loss_function(self, args):
        y_pred, labels, input_length, label_length = args
        # the 2 is critical here since the first couple outputs of the RNN
        # tend to be garbage:
        y_pred = y_pred[:, 2:, :]
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    def keras_model(self, is_training):

        # Make Networkw
        input_image = Input(
            name='the_input', shape=self._input_shape,
            dtype='float32')  # (None, 128, 64, 1)

        features = self._feature_extractor.extract_features(input_image)

        # CNN to RNN
        inner = Reshape(
            target_shape=((32, 2048)),
            name='reshape')(features)  # (None, 32, 2048)

        inner = Dense(
            64,
            activation='relu',
            kernel_initializer='he_normal',
            name='dense1')(inner)  # (None, 32, 64)

        # RNN layer
        lstm_1 = LSTM(
            256,
            return_sequences=True,
            kernel_initializer='he_normal',
            name='lstm1')(inner)  # (None, 32, 512)

        lstm_1b = LSTM(
            256,
            return_sequences=True,
            go_backwards=True,
            kernel_initializer='he_normal',
            name='lstm1_b')(inner)

        lstm1_merged = add([lstm_1, lstm_1b])  # (None, 32, 512)

        lstm1_merged = BatchNormalization()(lstm1_merged)

        lstm_2 = LSTM(
            256,
            return_sequences=True,
            kernel_initializer='he_normal',
            name='lstm2')(lstm1_merged)

        lstm_2b = LSTM(
            256,
            return_sequences=True,
            go_backwards=True,
            kernel_initializer='he_normal',
            name='lstm2_b')(lstm1_merged)

        lstm2_merged = concatenate([lstm_2, lstm_2b])  # (None, 32, 1024)

        lstm_merged = BatchNormalization()(lstm2_merged)

        # transforms RNN output to character activations:
        inner = Dense(
            num_classes, kernel_initializer='he_normal',
            name='dense2')(lstm2_merged)  #(None, 32, 63)

        y_pred = Activation('softmax', name='softmax')(inner)

        labels = Input(
            name='the_labels', shape=[max_text_len],
            dtype='float32')  # (None ,9)

        input_length = Input(
            name='input_length', shape=[1], dtype='int64')  # (None, 1)

        label_length = Input(
            name='label_length', shape=[1], dtype='int64')  # (None, 1)

        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer
        loss_out = Lambda(
            self.loss_function, output_shape=(1, ),
            name='ctc')([y_pred, labels, input_length,
                         label_length])  #(None, 1)

        if is_training:
            return Model(
                inputs=[input_image, labels, input_length, label_length],
                outputs=loss_out)
        else:
            return Model(inputs=[input_image], outputs=y_pred)
