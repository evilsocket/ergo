import os
import argparse
import logging as log

from ergo.dataset import Dataset

def probability(x):
    x = float(x)
    if x < 0 or x > 1:
        raise argparse.ArgumentTypeError("%r not in range [0.0 - 1.0]" % x )
    return x

def usage():
    print("usage: ergo optimize-dataset <filename> [args]")
    quit()

def parse_args(argv):
    parser = argparse.ArgumentParser(description="optimizer")
    parser.add_argument("-r", "--reuse-ratio", dest = "reuse", action = "store", type = probability, default = 0.15)
    parser.add_argument("-o", "--output", dest = "output", action = "store", type = str, default = None)
    args = parser.parse_args(argv)
    return args

def action_optimize_dataset(argc, argv):
    if argc < 1:
        usage()

    args = parse_args(argv[1:])
    path = os.path.abspath(argv[0])
    if not os.path.exists(path):
        log.error("dataset file %s does not exist", path)
        quit()

    Dataset.optimize(path, args.reuse, args.output) 
