import os
import argparse

from ergo.project import Project

def usage():
    print("usage: ergo clean <path> [args]")
    quit()

def parse_args(argv):
    parser = argparse.ArgumentParser(description="cleaner")
    parser.add_argument("-a", "--all", dest = "all", action = "store_true", default = False, help = "Remove model weights and training data.")
    args = parser.parse_args(argv)
    return args

def action_clean(argc, argv):
    if argc < 1:
        usage()

    args = parse_args(argv[1:])
    Project.clean(argv[0], args.all)
