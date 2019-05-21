import argparse
import json
import logging as log
import time
import numpy as np

from terminaltables import AsciiTable

from ergo.project import Project

# https://stackoverflow.com/questions/11942364/typeerror-integer-is-not-json-serializable-when-serializing-json-in-python
def default(o):
    if isinstance(o, np.int64): return int(o)
    raise TypeError

def validate_args(args):
    if args.ratio > 1.0 or args.ratio <= 0:
        log.error("ratio must be in the (0.0, 1.0] interval")
        quit()

def parse_args(argv):
    parser = argparse.ArgumentParser(prog="ergo relevance", description="Compute the relevance of each feature of the dataset by differential evaluation.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("path", help="Path of the project.")

    parser.add_argument("-d", "--dataset", dest="dataset", action="store", type=str, required=True,
        help="Dataset file to use.")
    parser.add_argument("-m", "--metric", dest="metric", action="store", type=str, default='precision',
        help="Which metric to track changes of while performing differential evaluation.")
    parser.add_argument("-a", "--attributes", dest="attributes", action="store", type=str, required=False,
        help="Optional file containing the attribute names, one per line.")
    parser.add_argument("-r", "--ratio", dest="ratio", action="store", type=float, required=False, default=1.0,
        help="Size of the subset of the dataset to use in the (0,1] interval.")
    parser.add_argument("-j", "--to-json", dest="to_json", action="store", type=str, required=False,
        help="Output the relevances to this json file.")

    args = parser.parse_args(argv)
    return args

def get_attributes(filename, ncols):
    attributes = []
    if filename is not None:
        with open(filename) as f:
            attributes = f.readlines()
        attributes = [name.strip() for name in attributes]
    else:
        attributes = ["feature %d" % i for i in range(0, ncols)]

    return attributes

def zeroize_feature(X, col, is_scalar_input):
    if is_scalar_input:
        backup = X[:, col].copy()
        X[:,col] = 0
    else:
        backup = X[col].copy()
        X[col] = np.zeros_like(backup)

    return backup

def restore_feature(X, col, backup, is_scalar_input):
    if is_scalar_input:
        X[:,col] = backup
    else:
        X[col] = backup

def action_relevance(argc, argv):
    args = parse_args(argv)
    prj = Project(args.path)
    err = prj.load()
    if err is not None:
        log.error("error while loading project: %s", err)
        quit()
    elif not prj.is_trained():
        log.error("no trained Keras model found for this project")
        quit()

    prj.prepare(args.dataset, 0.0, 0.0)

    X, y         = prj.dataset.subsample(args.ratio)
    nrows, ncols = X.shape if prj.dataset.is_flat else (X[0].shape[0], len(X))
    attributes   = get_attributes(args.attributes, ncols)

    log.info("computing relevance of %d attributes on %d samples using '%s' metric ...", ncols, nrows, args.metric)

    start = time.time()

    ref_accu, ref_cm = prj.accuracy_for(X, y, repo_as_dict = True)
    deltas           = []
    tot              = 0
    speed            = (1.0 / (time.time() - start)) * nrows

    for col in range(0, ncols):
        log.info("[%.2f evals/s] computing relevance for attribute [%d/%d] %s ...", speed, col + 1, ncols, attributes[col])

        backup = zeroize_feature(X, col, prj.dataset.is_flat)

        start = time.time()

        accu, cm = prj.accuracy_for(X, y, repo_as_dict = True)

        speed = (1.0 / (time.time() - start)) * nrows

        delta = ref_accu['weighted avg'][args.metric] - accu['weighted avg'][args.metric]
        tot  += delta

        deltas.append((col, delta))

        restore_feature(X, col, backup, prj.dataset.is_flat)

    deltas = sorted(deltas, key = lambda x: abs(x[1]), reverse = True)

    rels     = []
    num_zero = 0
    table    = [("Column", "Feature", "Relevance")]

    for delta in deltas:
        col, d = delta
        colname = attributes[col]
        rel = {
            "attribute": colname,
            "index": col, 
            "relevance": 0.0     
        }

        if d != 0.0:
            relevance = (d / tot) * 100.0
            row       = ("%d" % col, attributes[col], "%.2f%%" % relevance)
            row       = ["\033[31m%s\033[0m" % e for e in row] if relevance < 0.0 else row
            table.append(row)
            rel['relevance'] = relevance
        else:
            num_zero += 1

        rels.append(rel)

    print("")
    print(AsciiTable(table).table)
    print("")

    if num_zero > 0:
        log.info("%d features have 0 relevance.", num_zero)

    if args.to_json is not None:
        print("")
        log.info("creating %s ...", args.to_json)
        with open(args.to_json, 'w+') as fp:
            json.dump(rels, fp, default=default)
