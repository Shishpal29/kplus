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

from kplus.core.AbstractBatchGenerator import AbstractBatchGenerator


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

    train_dataset = AbstractBatchGenerator()
    status = train_dataset.load_train_dataset(parameters)
    if (status):
        print('The model is trained.')
    else:
        print('Error training the model.')


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
