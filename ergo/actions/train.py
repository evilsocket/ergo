import os
import argparse
import logging as log

from ergo.project import Project

def probability(x):
    x = float(x)
    if x < 0 or x > 1:
        raise argparse.ArgumentTypeError("%r not in range [0.0 - 1.0]" % x )
    return x

def usage():
    print("usage: ergo train <path> [args]")
    quit()

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

def parse_args(argv):
    parser = argparse.ArgumentParser(description="trainer")
    parser.add_argument("-d", "--dataset", dest = "dataset", action = "store", type = str, required = False)
    parser.add_argument("-g", "--gpus", dest = "gpus", action = "store", type = int, default = 0)
    parser.add_argument("-t", "--test", dest = "test", action = "store", type = probability, default = 0.15, required = False)
    parser.add_argument("-v", "--validation", dest = "validation", action = "store", type = probability, default = 0.15, required = False)
    parser.add_argument("--no-save", dest="no_save", action="store_true", default = False, help = "Do not save temporary datasets on disk.")
    parser.add_argument("--no-shuffle", dest="no_shuffle", action="store_true", default = False, help="Do not shuffle dataset during preparation.")
    args = parser.parse_args(argv)
    validate_args(args)
    return args

def action_train(argc, argv):
    if argc < 1:
        usage()

    prj = Project(argv[0])
    err = prj.load()
    if err is not None:
        log.error("error while loading project: %s", err)
        quit()

    args = parse_args(argv[1:])
    if args.dataset is not None:
        # a dataset was specified, split it and generate
        # the subsets
        prj.dataset.do_save = not args.no_save
        prj.prepare(args.dataset, args.test, args.validation, not args.no_shuffle)
    elif prj.dataset.exists():
        # no dataset passed, attempt to use the previously
        # generated subsets
        prj.dataset.load()
    else:
        log.error("no test/train/validation subsets found in %s, please specify a --dataset argument", argv[0])
        quit()

    prj.train(args.gpus)
