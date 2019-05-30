import argparse
import os
import logging as log

from ergo.project import Project
from ergo.core.utils import clean_if_exist

def probability(x):
    x = float(x)
    if x < 0 or x > 1:
        raise argparse.ArgumentTypeError("%r not in range [0.0 - 1.0]" % x)
    return x


def validate_args(args):
    if args.dataset is not None and (not args.dataset.startswith('sum://') and not os.path.exists(args.dataset)):
        log.error("file %s does not exist" % args.dataset)
        quit()

    p_train = 1 - args.test - args.validation
    if p_train < 0:
        log.error("validation and test proportion are bigger than 1")
        quit()
    elif p_train < 0.5:
        log.error("using less than 50% of the dataset for training")
        quit()

def clean_dataset(path):
    files = [ \
        'data-train.csv',
        'data-test.csv',
        'data-validation.csv',
        'data-train.pkl',
        'data-test.pkl',
        'data-validation.pkl']

    clean_if_exist(path, files)

def parse_args(argv):
    parser = argparse.ArgumentParser(prog="ergo prepare", description="Create train, test and validation sets.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("path", help="The path containing the model definition.")

    parser.add_argument("-d", "--dataset", action="store", dest="dataset",
                        help="Path of the dataset to use or a SUM server address. Leave empty to reuse previously generated subsets.")
    parser.add_argument("-t", "--test", action="store", dest="test", type=probability, default=0.15,
                        help="Proportion of the test set in the (0,1] interval.")
    parser.add_argument("-v", "--validation", action="store", dest="validation", type=probability, default=0.15,
                        help="Proportion of the validation set in the (0,1] interval.")
    parser.add_argument("--no-shuffle", dest="no_shuffle", action="store_true", default=False,
                        help="Do not shuffle dataset during preparation.")

    args = parser.parse_args(argv)
    validate_args(args)
    return args


def action_prepare(argc, argv):
    args = parse_args(argv)
    prj = Project(args.path)
    err = prj.load()
    if err is not None:
        log.error("error while loading project: %s", err)
        quit()

    if args.dataset is None:
        log.error("no --dataset argument specified", args.path)
        quit()

    if prj.dataset.exists():
        log.info("removing previously generated datasets")
        clean_dataset(args.path)

    prj.dataset.do_save = True
    prj.prepare(args.dataset, args.test, args.validation, not args.no_shuffle)