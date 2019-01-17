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
        self._feature_extractor = ModelFactory.simple_model(feature_extractor)
        self._feature_extractor.build(input_shape=self._input_shape)

    def _loss_function(self, inputs):
        input_labels, predicted_output, input_length, label_length = inputs
        # the 2 is critical here since the first couple outputs of the RNN
        # tend to be garbage:
        predicted_output = predicted_output[:, 2:, :]
        return (K.ctc_batch_cost(input_labels, predicted_output, input_length,
                                 label_length))

    def _encode_sequence(self, layer_input):
        # RNN encoder layer
        forward_lstm = LSTM(
            256,
            return_sequences=True,
            kernel_initializer='he_normal',
            name='forward_lstm_1')(layer_input)  # (None, 32, 512)

        backward_lstm = LSTM(
            256,
            return_sequences=True,
            go_backwards=True,
            kernel_initializer='he_normal',
            name='backward_lstm_1')(layer_input)

        merged_lstm = add([forward_lstm, backward_lstm])  # (None, 32, 512)
        merged_lstm = BatchNormalization()(merged_lstm)

        layer_output = merged_lstm

        return (layer_output)

    def keras_model(self, is_training):

        input_image = Input(
            name='input_image', shape=self._input_shape,
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

        encoded_sequence = self._encode_sequence(inner)

        lstm_2 = LSTM(
            256,
            return_sequences=True,
            kernel_initializer='he_normal',
            name='lstm2')(encoded_sequence)

        lstm_2b = LSTM(
            256,
            return_sequences=True,
            go_backwards=True,
            kernel_initializer='he_normal',
            name='lstm2_b')(encoded_sequence)

        lstm2_merged = concatenate([lstm_2, lstm_2b])  # (None, 32, 1024)

        lstm_merged = BatchNormalization()(lstm2_merged)

        # RNN to output prediction
        inner = Dense(
            num_classes, kernel_initializer='he_normal',
            name='dense2')(lstm2_merged)  #(None, 32, 63)

        predicted_output = Activation('softmax', name='softmax')(inner)

        input_labels = Input(
            name='input_labels', shape=[max_text_len],
            dtype='float32')  # (None ,9)

        input_length = Input(
            name='input_length', shape=[1], dtype='int64')  # (None, 1)

        label_length = Input(
            name='label_length', shape=[1], dtype='int64')  # (None, 1)

        output_loss = Lambda(
            self._loss_function, output_shape=(1, ), name='ctc')(
                [input_labels, predicted_output, input_length,
                 label_length])  #(None, 1)

        if (is_training):
            return Model(
                inputs=[input_image, input_labels, input_length, label_length],
                outputs=output_loss)
        else:
            return Model(inputs=[input_image], outputs=predicted_output)
