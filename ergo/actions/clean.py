import os
import argparse

from ergo.project import Project

def parse_args(argv):
    parser = argparse.ArgumentParser(prog="ergo clean", description="Clean a project from temporary datasets and optionally reset it to its initial state.")

    parser.add_argument("path", help="Path of the project to clean.")
    parser.add_argument( "-a", "--all", dest="all", action="store_true", default=False,
        help="Remove model weights and training data.")

    args = parser.parse_args(argv)
    return args

def action_clean(argc, argv):
    args = parse_args(argv)
    Project.clean(args.path, args.all)
