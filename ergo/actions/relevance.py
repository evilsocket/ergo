import os
import json
import argparse
import logging as log
import numpy as np

from terminaltables import AsciiTable

from ergo.project import Project

def usage():
    print("usage: ergo relevance <path> --dataset <path> --attributes <path>")
    quit()

def parse_args(argv):
    parser = argparse.ArgumentParser(description="relevance")
    parser.add_argument("-d", "--dataset", dest = "dataset", action = "store", type = str, required = True)
    parser.add_argument("-a", "--attributes", dest = "attributes", action = "store", type = str, required = False)
    args = parser.parse_args(argv)
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

    nrows, ncols = prj.dataset.X.shape

    attributes = []

    if args.attributes is not None:
        with open(args.attributes) as f:
            attributes = f.readlines()
        attributes = [x.strip() for x in attributes] 
    else:
        attributes = ["feature %d" % x for x in range(0, ncols)]

    log.info("computing relevance of %d attributes on %d samples ...", ncols, nrows)

    ref_accu, ref_cm = prj.accuracy_for(prj.dataset.X, prj.dataset.Y, repo_as_dict = True)
    deltas = []
    tot = 0

    for col in range(0, ncols):
        log.info("computing relevance for attribute [%d/%d] %s ...", col + 1, ncols, attributes[col])

        backup = prj.dataset.X[:,col].copy()

        prj.dataset.X[:,col] = 0

        accu, cm = prj.accuracy_for(prj.dataset.X, prj.dataset.Y, repo_as_dict = True)

        delta = ref_accu['weighted avg']['precision'] - accu['weighted avg']['precision']
        tot += delta

        deltas.append((col, delta))

        prj.dataset.X[:,col] = backup

    deltas = sorted(deltas, key = lambda x: x[1], reverse = True)

    num_irrelevant = 0
    table = [("Feature", "Relevance")]
    for delta in deltas:
        col, d = delta
        relevance = (d / tot) * 100.0
        if relevance > 0:
            table.append((attributes[col], "%.2f%%" % relevance))
        else:
            num_irrelevant += 1

    print("")
    print(AsciiTable(table).table)
    print("")

    if num_irrelevant > 0:
        log.info("%d features have 0 relevance.", num_irrelevant)
