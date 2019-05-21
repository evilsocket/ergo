import os
import argparse

from ergo.core.utils import clean_if_exist

def parse_args(argv):
    parser = argparse.ArgumentParser(prog="ergo clean", description="Clean a project from temporary datasets and optionally reset it to its initial state.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("path", help="Path of the project to clean.")
    parser.add_argument( "-a", "--all", dest="all", action="store_true", default=False,
        help="Remove model weights and training data.")

    args = parser.parse_args(argv)
    return args

def action_clean(argc, argv):
    args = parse_args(argv)
    files = [ \
        'data-train.csv',
        'data-test.csv',
        'data-validation.csv',
        'data-train.pkl',
        'data-test.pkl',
        'data-validation.pkl']

    if args.all:
        files += [ \
            '__pycache__',
            'logs',
            'model.yml',
            'model.h5',
            'model.fdeep',
            'model.stats', # legacy
            'model.png', # legacy
            'test_cm.png',
            'training_cm.png',
            'validation_cm.png',
            'history.png',
            'roc.png',
            'stats.txt',
            'stats.json',
            'history.json']
    
    clean_if_exist(args.path, files)
