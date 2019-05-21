import os
import argparse
import logging as log

from ergo.project import Project

def parse_args(argv):
    parser = argparse.ArgumentParser(prog="ergo view", description="View the model struture and training statistics.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("path", help="Path of the project.")
    parser.add_argument( "--img-only", dest="img_only", default=False, action="store_true",
        help="Save plots as PNG files but don't show them in a UI.")
    args = parser.parse_args(argv)
    return args

def action_view(argc, argv):
    args = parse_args(argv)
    prj = Project(args.path)
    err = prj.load()
    if err is not None:
        log.error("error while loading project: %s", err)
        quit()

    prj.view(args.img_only)
