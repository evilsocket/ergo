import os
import argparse
import logging as log

from ergo.project import Project

def probability(x):
    x = float(x)
    if x < 0 or x > 1:
        raise argparse.ArgumentTypeError("%r not in range [0.0 - 1.0]" % x )
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

def parse_args(argv):
    parser = argparse.ArgumentParser(prog="ergo train", description="Start the training phase of a model.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("path", help="The path containing the model definition.")

    parser.add_argument( "-d", "--dataset", action="store", dest="dataset", 
        help="Path of the dataset to use or a SUM server address. Leave empty to reuse previously generated subsets.")
    parser.add_argument("-g", "--gpus", action="store", dest="gpus", type=int, default=0, 
        help="Number of GPU devices to use, leave to 0 to use all the available ones.")
    parser.add_argument("-t", "--test", action="store", dest="test", type=probability, default=0.15, 
        help="Proportion of the test set in the (0,1] interval.")
    parser.add_argument("-v", "--validation", action="store", dest="validation", type=probability, default=0.15, 
        help="Proportion of the validation set in the (0,1] interval.")
    
    parser.add_argument("--no-save", dest="no_save", action="store_true", default=False, 
        help="Do not save temporary datasets on disk.")
    parser.add_argument("--no-shuffle", dest="no_shuffle", action="store_true", default=False, 
        help="Do not shuffle dataset during preparation.")

    args = parser.parse_args(argv)
    validate_args(args)
    return args

def action_train(argc, argv):
    args = parse_args(argv)
    prj = Project(args.path)
    err = prj.load()
    if err is not None:
        log.error("error while loading project: %s", err)
        quit()

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
        log.error("no test/train/validation subsets found in %s, please specify a --dataset argument", args.path)
        quit()

    prj.train(args.gpus)
