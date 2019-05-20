import sys
import os
import argparse
import logging as log

from ergo.project import Project

def parse_args(argv):
    parser = argparse.ArgumentParser(prog="ergo create", description="Create a new ergo project.")
    parser.add_argument("path", help="Path of the project to create.")
    
    parser.add_argument("-i", "--inputs", dest="num_inputs", action="store", type=int, default=10,
        help="Number of inputs of the model.")
    parser.add_argument("-o", "--outputs", dest="num_outputs", action="store", type=int, default=2,
        help="Number of outputs of the model.")
    parser.add_argument("-l", "--layers", dest="hidden", action="store", type=str, default="30, 30",
        help="Comma separated list of positive integers, one per each hidden layer representing its size.")
    parser.add_argument("-b", "--batch-size", dest="batch_size", action="store", type=int, default=64,
        help="Batch size parameter for training.")
    parser.add_argument("-e", "--epochs", dest="max_epochs", action="store", type=int, default=50,
        help="Maximum number of epochs to train the model.")

    args = parser.parse_args(argv)
    return args

def action_create(argc, argv):
    args = parse_args(argv)
    if os.path.exists(args.path):
        log.error("path %s already exists" % args.path)
        quit()

    Project.create(args.path, args.num_inputs, args.hidden, args.num_outputs, args.batch_size, args.max_epochs)




