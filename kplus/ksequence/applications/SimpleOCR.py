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

from keras import backend as K
from keras.optimizers import Adadelta
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from kplus.ksequence.datasets.SimpleGenerator import SimpleGenerator
from kplus.ksequence.models.ModelFactory import ModelFactory
from kplus.ksequence.parameter import *

from kplus.core.AbstractApplication import AbstractApplication


class SimpleOCR(AbstractApplication):
    def __init__(self):
        pass

    def _setup_early_stop(self):
        pass

    def _setup_model_checkpoint(self):
        pass

    def _setup_tensorboard(self):
        pass

    def _change_learning_rate(self):
        pass

    def train(self, parameters):

        model_name = parameters['model']['model_name']
        if (not (model_name in ['base', 'bidirectional', 'attention'])):
            return (False)

        feature_extractor = parameters['model']['feature_extractor']
        if (not (feature_extractor in ['simple_vgg', 'resnet_50'])):
            return (False)

        train_file_path = parameters['train_dataset_dir']
        valid_file_path = parameters['test_dataset_dir']
        epoch = parameters['max_number_of_epoch']

        sequence_model = ModelFactory.simple_model(model_name)
        sequence_model.use_feature_extractor(feature_extractor)
        keras_model = sequence_model.keras_model(is_training=True)

        try:
            keras_model.load_weights('LSTM+BN4--26--0.011.hdf5')
            print("...Previous weight data...")
        except:
            print("...New weight data...")
            pass

        tiger_train = SimpleGenerator(train_file_path, img_w, img_h,
                                      batch_size, downsample_factor)
        tiger_train.build_data()

        tiger_val = SimpleGenerator(valid_file_path, img_w, img_h,
                                    val_batch_size, downsample_factor)
        tiger_val.build_data()

        ada = Adadelta()

        # Dummy lambda function for the loss
        keras_model.compile(
            loss={
                'ctc': lambda y_true, y_pred: y_pred
            },
            optimizer=ada,
            metrics=['accuracy'])

        early_stop = EarlyStopping(
            monitor='loss', min_delta=0.001, patience=4, mode='min', verbose=1)

        train_root_dir = os.path.expanduser(parameters['train_root_dir'])
        checkpoint_path = os.path.join(
            train_root_dir, 'model--{epoch:03d}--{val_loss:.5f}.hdf5')
        tensorboard_path = os.path.join(train_root_dir, 'tensorboard')

        checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='loss',
            verbose=1,
            mode='min',
            period=1)

        tensor_board = TensorBoard(
            log_dir=tensorboard_path,
            histogram_freq=0,
            write_graph=True,
            write_images=False)

        keras_model.fit_generator(
            generator=tiger_train.next_batch(),
            steps_per_epoch=int(tiger_train.n / batch_size),
            callbacks=[checkpoint, early_stop, tensor_board],
            epochs=epoch,
            validation_data=tiger_val.next_batch(),
            validation_steps=int(tiger_val.n / val_batch_size))

        return (True)

    def evaluate(self, parameters):
        pass

    def predict(self, input_image):
        pass
