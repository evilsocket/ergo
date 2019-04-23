import argparse
import logging as log
import time
import numpy as np

from terminaltables import AsciiTable

from ergo.project import Project

def usage():
    print("usage: ergo relevance <path> --dataset <path> --attributes <path>")
    quit()

def validate_args(args):
    if args.ratio > 1.0 or args.ratio <= 0:
        log.error("ratio must be in the (0.0, 1.0] interval")
        quit()

def parse_args(argv):
    parser = argparse.ArgumentParser(description="relevance")
    parser.add_argument("-d", "--dataset", dest = "dataset", action = "store", type = str, required = True)
    parser.add_argument("-a", "--attributes", dest = "attributes", action = "store", type = str, required = False)
    parser.add_argument("-r", "--ratio", dest = "ratio", action = "store", type = float, required = False, default = 1.0)
    args = parser.parse_args(argv)
    validate_args(args)
    return args

def action_relevance(argc, argv):
    if argc < 3:
        usage()

    args = parse_args(argv[1:])
    prj = Project(argv[0])
    err = prj.load()
    if err is not None:
        log.error("error while loading project: %s", err)
        quit()
    elif not prj.is_trained():
        log.error("no trained Keras model found for this project")
        quit()

    prj.prepare(args.dataset, 0.0, 0.0)

    X = np.matrix(prj.dataset.X)
    y = np.array(prj.dataset.Y)

    num = int(X.shape[0] * args.ratio)

    if args.ratio < 1.0:
        log.info("selecting a randomized sample of %d%% ...", args.ratio * 100)
        indexes = np.random.choice(X.shape[0], num, replace = False)
        X = X[indexes]
        y = y[indexes]

    nrows, ncols = X.shape
    attributes = []

    if args.attributes is not None:
        with open(args.attributes) as f:
            attributes = f.readlines()
        attributes = [name.strip() for name in attributes] 
    else:
        attributes = ["feature %d" % i for i in range(0, ncols)]

    log.info("computing relevance of %d attributes on %d samples ...", ncols, nrows)

    ref_accu, ref_cm = prj.accuracy_for(X, y, repo_as_dict = True)
    deltas = []
    tot = 0
    speed = 0.0

    for col in range(0, ncols):
        log.info("[%.2f evals/s] computing relevance for attribute [%d/%d] %s ...", speed, col + 1, ncols, attributes[col])

        backup_w, backup_b = prj.null_feature(col)

        start = time.time()

        accu, cm = prj.accuracy_for(X, y, repo_as_dict = True)

        end = time.time()

        speed = 1.0 / (end - start)

        delta = ref_accu['weighted avg']['precision'] - accu['weighted avg']['precision']
        tot += delta

        deltas.append((col, delta))

        prj.restore_feature(col, backup_w, backup_b)

    deltas = sorted(deltas, key = lambda x: abs(x[1]), reverse = True)

    num_irrelevant = 0
    table = [("Feature", "Relevance")]
    for delta in deltas:
        col, d = delta
        if d != 0.0:
            relevance = (d / tot) * 100.0
            table.append((attributes[col], "%.2f%%" % relevance))
        else:
            num_irrelevant += 1

    print("")
    print(AsciiTable(table).table)
    print("")

    if num_irrelevant > 0:
        log.info("%d features have 0 relevance.", num_irrelevant)
