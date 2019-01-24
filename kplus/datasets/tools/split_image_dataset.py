import os, random
import shutil
import argparse
import sys


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-source",
        "--source_dir",
        help="source data directory",
        type=str,
        default="data/")
    parser.add_argument(
        "-target",
        "--target_dir",
        help="target data directory",
        type=str,
        default="../datasets/")
    parser.add_argument(
        "-test", "--test_ratio", help="test data ratio", type=int, default=5)
    parser.add_argument(
        "-val",
        "--val_ratio",
        help="validation data ratio",
        type=int,
        default=5)

    return (parser.parse_args(argv))


def build(source_dir, target_dir, test_ratio, val_ratio):
    train_ratio = 100 - test_ratio - val_ratio

    assert (os.path.exists(source_dir)), "Invalid source directory path"

    target_dir = os.path.expanduser(target_dir)
    source_dir = os.path.expanduser(source_dir)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    os.makedirs(os.path.join(target_dir, "train/"))
    os.makedirs(os.path.join(target_dir, "test/"))
    os.makedirs(os.path.join(target_dir, "val/"))

    train_path = os.path.join(target_dir, "train/")
    test_path = os.path.join(target_dir, "test/")
    val_path = os.path.join(target_dir, "val/")

    filenames = os.listdir(source_dir)

    filenames.sort(
    )  # make sure that the filenames have a fixed order before shuffling

    random.seed(230)
    random.shuffle(
        filenames
    )  # shuffles the ordering of filenames (deterministic given the chosen seed)

    split_1 = int((train_ratio / 100) * len(filenames))
    split_2 = int(((train_ratio + val_ratio) / 100) * len(filenames))

    split = 0
    for f in filenames:
        src = os.path.join(source_dir, f)
        if (split < split_1):
            dst = os.path.join(train_path, f)
        elif (split < split_2):
            dst = os.path.join(val_path, f)
        else:
            dst = os.path.join(test_path, f)
        shutil.move(src, dst)
        split += 1


def main(args):
    source_dir = args.source_dir
    target_dir = args.target_dir
    test_ratio = args.test_ratio
    val_ratio = args.val_ratio

    build(source_dir, target_dir, test_ratio, val_ratio)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
