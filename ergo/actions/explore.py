import argparse
from terminaltables import AsciiTable
import logging as log
import numpy as np
import os

from ergo.project import Project
import ergo.views as views

def validate_ratio(args):
    if args.ratio > 1.0 or args.ratio <= 0:
        log.error("ratio must be in the (0.0, 1.0] interval")
        quit()

def parse_args(argv):
    parser = argparse.ArgumentParser(prog="ergo explore", description="Explore dataset properties",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("path", help="Path of the project to explore.")
    parser.add_argument("-a", "--attributes", dest="attributes", action="store", type=str, required=False,
                        help="Optional file containing the attribute names, one per line.")
    parser.add_argument("-r", "--ratio", dest="ratio", action="store", type=validate_ratio, required=False, default=1.0,
                        help="Size of the subset of the dataset to use in the (0,1] interval.")
    parser.add_argument("-d", "--dataset", dest="dataset", action="store", type=str, required=True,
                        help="Dataset file to use.")
    return parser.parse_args(argv)

def get_attributes(filename, ncols):
    attributes = []
    if filename is not None:
        with open(filename) as f:
            attributes = f.readlines()
        attributes = [name.strip() for name in attributes]
    else:
        attributes = ["feature %d" % i for i in range(0, ncols)]

    return attributes

prj = None
ncols = 0
nrows = 0
attributes = None


def compute_correlations_with_target (X,y):
    global nrows, ncols, attributes
    y = np.argmax(y, axis=1)
    corr = [(attributes[i], np.corrcoef(X[:, i], y)[0,1], i)
            for i in range(0, ncols)]
    corr = [(i, j, k) for i, j, k in corr if not np.isnan(j)]
    corr = sorted(corr, key=lambda x: abs(x[1]), reverse=True)
    return corr


def print_correlation_table(corr, min_corr = 0.3):
    table = [("Column", "Feature", "Correlation")]
    not_relevant = 0
    for at, c, idx in corr:
        if abs(c) < min_corr:
            not_relevant += 1
            continue
        row = ( "%d" % idx, "%s" % at, "%.4f" % c)
        table.append(row)

    print("")
    print(AsciiTable(table).table)
    print("")

    if not_relevant > 0:
        log.info("%d features have correlation lower than %f with target attribute.", not_relevant, min_corr)

def calculate_pca(X):
    from sklearn.decomposition import PCA
    dec = PCA()
    dec.fit(X)
    return dec


def action_explore(argc, argv):
    global prj, nrows, ncols, attributes

    args     = parse_args(argv)
    prj = Project(args.path)
    err = prj.load()
    if err is not None:
        log.error("error while loading project: %s", err)
        quit()

    prj.prepare(args.dataset, 0.0, 0.0)
    if not prj.dataset.is_flat:
        log.error("data exploration can only be applied to flat inputs")
        quit()

    X, y = prj.dataset.subsample(args.ratio)
    nrows, ncols = X.shape
    attributes = get_attributes(args.attributes, ncols)
    corr = compute_correlations_with_target(X,y)
    print_correlation_table(corr)
    pca = calculate_pca(X)
    views.pca_projection(prj, pca, X, y, False)
    views.pca_explained_variance(prj, pca, False)
    views.show(False)
