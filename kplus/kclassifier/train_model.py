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

import sys
import argparse
import json

from kplus.kclassifier.datasets.ClassifierBatchGenerator import ClassifierBatchGenerator

from keras.layers import Input
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Model
from keras.applications.resnet50 import ResNet50


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--parameter_filename',
        type=str,
        help='Input parameter file name.',
        default='./parameters/parameters.json')

    return (parser.parse_args(argv))


def main(args):
    parameter_filename = args.parameter_filename

    with open(parameter_filename) as input_buffer:
        parameters = json.loads(input_buffer.read())

    status = True

    train_dataset = ClassifierBatchGenerator()
    status = train_dataset.load_train_dataset(parameters) and status

    val_dataset = ClassifierBatchGenerator()
    status = val_dataset.load_val_dataset(parameters) and status

    test_dataset = ClassifierBatchGenerator()
    status = test_dataset.load_test_dataset(parameters) and status
    #X, y = test_dataset[1]
    #print(y)

    input_layer = Input(shape=(224, 224, 3))
    base_model = ResNet50(
        weights=None, include_top=False, input_tensor=input_layer)

    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(5, activation='softmax')(x)

    model = Model(input=base_model.input, output=predictions)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy'])

    checkpoint = ModelCheckpoint(
        filepath='./train/', monitor='loss', verbose=1, mode='auto', period=1)

    model.fit_generator(
        generator=train_dataset,
        steps_per_epoch=train_dataset.steps_per_epoch(),
        callbacks=[checkpoint],
        epochs=5,
        validation_data=val_dataset,
        validation_steps=val_dataset.steps_per_epoch())

    if (status):
        print('The model is trained.')
    else:
        print('Error training the model.')


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
