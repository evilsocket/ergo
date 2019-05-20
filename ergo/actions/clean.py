import os
import argparse

from ergo.core.utils import clean_if_exist

from ergo.project import Project
from ergo.dataset import Dataset

def parse_args(argv):
    parser = argparse.ArgumentParser(prog="ergo clean", description="Clean a project from temporary datasets and optionally reset it to its initial state.")

    parser.add_argument("path", help="Path of the project to clean.")
    parser.add_argument( "-a", "--all", dest="all", action="store_true", default=False,
        help="Remove model weights and training data.")

    args = parser.parse_args(argv)
    return args

def action_clean(argc, argv):
    args = parse_args(argv)

    Dataset.clean(args.path)
    if args.all:
        # clean everything
        clean_if_exist(args.path, ( \
            '__pycache__',
            'logs',
            'model.yml',
            'model.h5',
            'model.fdeep',
            'model.stats', # legacy
            'model.png', # legacy
            'test_cm.png',
            'train_cm.png',
            'validation_cm.png',
            'history.png',
            'roc.png',
            'stats.txt',
            'stats.json',
            'history.json'))
