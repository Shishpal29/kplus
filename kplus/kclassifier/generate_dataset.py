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

from kplus.core.AbstractBatchGenerator import AbstractBatchGenerator


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--source_root_dir',
        type=str,
        help='Input source directory.',
        default='./source')

    parser.add_argument(
        '--target_root_dir',
        type=str,
        help='Output directory.',
        default='./target')

    return (parser.parse_args(argv))


def main(args):

    if (not args.source_root_dir):
        raise ValueError(
            'You must supply input source directory with --source_root_dir.')

    if (not args.target_root_dir):
        raise ValueError(
            'You must supply output directory with --target_root_dir.')

    batch_generator = AbstractBatchGenerator()
    status = batch_generator.generate_dataset(args.source_root_dir,
                                              args.target_root_dir)
    if (status):
        print('Dataset is generated at ' + args.target_root_dir)
    else:
        print('Error generating the dataset.')


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
