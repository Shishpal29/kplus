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
import random
import fnmatch
import argparse
import sys


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--source_root_dir',
        type=str,
        help='Input source directory.',
        default='/datasets/flowers/')

    parser.add_argument(
        '--target_root_dir',
        type=str,
        help='Output directory.',
        default='/datasets/flowers/')

    parser.add_argument(
        "--validation_ratio",
        help="Validation data ratio",
        type=float,
        default=5.0)

    parser.add_argument(
        "--test_ratio", help="Test data ratio.", type=float, default=5.0)

    return (parser.parse_args(argv))


def main(args):

    if (not args.source_root_dir):
        raise ValueError(
            'You must supply source root directory with --source_root_dir.')

    if (not args.target_root_dir):
        raise ValueError(
            'You must supply target root directory with --target_root_dir.')

    if (not os.path.exists(args.source_root_dir)):
        return (False)

    test_ratio = args.test_ratio
    if (args.test_ratio < 0.0):
        test_ratio = 0.0

    validation_ratio = args.validation_ratio
    if (args.validation_ratio < 0.0):
        validation_ratio = 0.0

    if ((test_ratio + validation_ratio) > 100.0):
        test_ratio = validation_ratio = 0.0

    train_ratio = 100.0 - (test_ratio + validation_ratio)

    target_root_dir = os.path.expanduser(args.target_root_dir)
    source_root_dir = os.path.expanduser(args.source_root_dir)

    if (not os.path.exists(target_root_dir)):
        os.makedirs(target_root_dir)

    train_root_dir = os.path.join(target_root_dir, "train")
    test_root_dir = os.path.join(target_root_dir, "test")
    validation_root_dir = os.path.join(target_root_dir, "val")

    os.makedirs(train_root_dir)
    os.makedirs(test_root_dir)
    os.makedirs(validation_root_dir)

    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]

    class_names = [
        class_name for class_name in os.listdir(source_root_dir)
        if os.path.isdir(os.path.join(source_root_dir, class_name))
    ]

    for class_name in class_names:
        class_source_dir = os.path.join(source_root_dir, class_name)

        class_target_train_dir = os.path.join(target_root_dir, "train",
                                              class_name)
        os.makedirs(class_target_train_dir)

        class_target_val_dir = os.path.join(target_root_dir, "val", class_name)
        os.makedirs(class_target_val_dir)

        class_target_test_dir = os.path.join(target_root_dir, "test",
                                             class_name)
        os.makedirs(class_target_test_dir)

        image_filenames = []
        for pattern in patterns:
            current_pattern_images = fnmatch.filter(
                os.listdir(class_source_dir), pattern)
            current_images = len(current_pattern_images)
            if (current_images > 0):
                image_filenames = image_filenames + current_pattern_images

        random.shuffle(image_filenames)
        total_images = len(image_filenames)

        training_images = int((train_ratio / 100.0) * total_images)
        validation_images = int((validation_ratio / 100.0) * total_images)

        current_image = 0
        for image_filename in image_filenames:
            source_filename = os.path.join(class_source_dir, image_filename)

            if (current_image < training_images):
                target_filename = os.path.join(class_target_train_dir,
                                               image_filename)
            elif (current_image < (training_images + validation_images)):
                target_filename = os.path.join(class_target_val_dir,
                                               image_filename)
            else:
                target_filename = os.path.join(class_target_test_dir,
                                               image_filename)

            os.rename(source_filename, target_filename)
            current_image = current_image + 1

    return (True)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
