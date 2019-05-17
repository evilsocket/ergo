import sys
import os
import argparse
import logging as log

from ergo.project import Project

def parse_args(argv):
    parser = argparse.ArgumentParser(prog="ergo create", description="Create a new ergo project.")
    parser.add_argument("path", help="Path of the project to create.")
    args = parser.parse_args(argv)
    return args

def action_create(argc, argv):
    args = parse_args(argv)
    if os.path.exists(args.path):
        log.error("path %s already exists" % args.path)
        quit()

    Project.create(args.path)




