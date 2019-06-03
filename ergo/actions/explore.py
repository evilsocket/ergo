import argparse
from terminaltables import AsciiTable
import logging as log
import numpy as np
import pandas as pd
np.seterr(divide='ignore', invalid='ignore')

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
    parser.add_argument("--img-only", dest="img_only", default=False, action="store_true",
                        help="Save plots as PNG files but don't show them in a UI.")
    parser.add_argument("-s", dest="stats", action="store_true", default=False,
                        help="Show dataset statistics")
    parser.add_argument("-c", dest="correlations", action="store_true", default=False,
                        help="Show correlation tables and graphs")
    parser.add_argument("-p", dest="pca", action="store_true", default=False,
                        help="Calculate pca decomposition and explained variance")
    parser.add_argument("-k", dest="cluster", action="store_true", default=False,
                        help="Perform clustering analysis")
    parser.add_argument("--algorithm", dest="cluster_alg", choices=['kmeans', 'dbscan'], default='kmeans', const='kmeans', nargs='?',
                        help="Algorithm for clustering analysis")
    parser.add_argument("-n", dest="nclusters", action="store", required = False, type=float,
                        help="Number of clusters for Kmeans algorithm")
    parser.add_argument("--nmax", dest="nmaxclusters", action="store", required=False, type=int,
                        help="Perform inertia analysis using nmax clusters at most")
    parser.add_argument("--3D", dest="D3", action="store_true", default=False,
                        help="Plot 3D projections of the data")
    parser.add_argument("--all", dest="all", action="store_true", default=False,
                        help="Process all capabilities of ergo explore (can be time consuming deppending on dataset")
    parser.add_argument("-w", "--workers", dest="workers", action="store", type=int, default=0,
                        help="If 0, all the algorithms will run a single thread. Otherwise all algorithms will use w number of jobs "+ \
                             "to distribute the computation among, -1 to use the number of logical CPU cores available.")
    return parser.parse_args(argv)

def red(s):
    return "\033[31m" + s + "\033[0m"

def terminal(s):
    return s

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
n_jobs = 1


def compute_correlations_with_target (X,y):
    global nrows, ncols, attributes
    y = np.argmax(y, axis=1)
    corr = [(attributes[i], np.corrcoef(X[:, i], y)[0,1], i)
            for i in range(0, ncols)]
    corr = [(i, j, k) for i, j, k in corr if not np.isnan(j)]
    corr = sorted(corr, key=lambda x: abs(x[1]), reverse=True)
    return corr


def print_target_correlation_table(corr, min_corr = 0.3):
    table = [("Column", "Feature", "Correlation with target")]
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


def calculate_corr(X):
    global attributes
    return pd.DataFrame(np.corrcoef(X, rowvar=False), columns=attributes, index=attributes)


def print_correlation_table(corr, min_corr=0.6):
    abs_corr = corr.abs().unstack()
    sorted_corr = abs_corr.sort_values(kind = "quicksort", ascending=False)
    table = [("Feature", "Feature", "Correlation")]
    for idx in sorted_corr.index:
        if idx[0] == idx[1]:
            continue
        if sorted_corr.loc[idx] <= min_corr:
            break
        if (idx[1], idx[0], sorted_corr.loc[idx]) in table:
            continue
        table.append(( idx[0], idx[1], "%.3f" % sorted_corr.loc[idx]))
    print("")
    log.info("Showing variables with a correlation higher than %f" % min_corr)
    print("")
    print(AsciiTable(table).table)
    print("")


def print_stats_table(X):
    global attributes
    minv = X.min(axis = 0)
    maxv = X.max(axis = 0)
    stdv = X.std(axis = 0)

    table = [("Feature", "Min value", "Max value", "Standard deviation")]
    constant = []
    unormalized = []
    for a, mi, ma, st in zip(attributes, minv, maxv, stdv ):
        color = terminal
        if mi == ma:
            constant.append(a)
            continue
        if mi < -1 or ma > 1:
            color = red
            unormalized.append(color(a))

        table.append( (color(a), color("%.2e" % mi) , color("%.2e" % ma), color( "%.2e"% st)))

    print("")
    log.info("Features distribution:")
    print("")
    print(AsciiTable(table).table)
    print("")
    log.warning("The following attributes have constant values: %s" % ', '.join(constant))
    print("")
    log.warning("The following features are not normalized: %s" % ', '.join(unormalized))


def kmeans_clustering(X, n_clusters):
    global n_jobs
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters = n_clusters, n_jobs=n_jobs)
    km.fit(X)
    return km


def dbscan_clustering(X, var):
    global n_jobs
    from sklearn.cluster import DBSCAN
    db = DBSCAN(var, min_samples=10, n_jobs=n_jobs)
    db.fit(X)
    return db


def n_clusters_analysis(X, nmax = 10, nmin = 2):
    global prj, n_jobs
    from sklearn.cluster import KMeans

    inertia = []
    for n in range(nmin, nmax + 1):
        log.info("fitting %d clusters to data" % n)
        km = KMeans(n_clusters = n, n_jobs=n_jobs).fit(X)
        inertia.append(km.inertia_)
    views.plot_intertia(prj, range(nmin, nmax + 1), inertia)


def action_explore(argc, argv):
    global prj, nrows, ncols, attributes, n_jobs

    args     = parse_args(argv)

    if args.all:
        args.pca = True
        args.correlations = True
        args.stats = True
        args.cluster = True
        args.D3 = True

    if args.workers == -1:
        import multiprocessing
        n_jobs = multiprocessing.cpu_count()
    elif args.workers != 0:
        n_jobs = args.workers
    log.info("using %d workers" % n_jobs)

    if args.nclusters and not args.cluster:
        log.warning("number of clusters specified but clustering won't be perfomed")

    if not (args.pca or args.correlations or args.stats or args.cluster):
        log.error("No exploration action was specified")
        print("")
        parse_args(["-h"])
        quit()

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

    if args.correlations:
        log.info("computing correlations of each feature with target")
        corr = compute_correlations_with_target(X,y)
        print_target_correlation_table(corr)
        log.info("computing features crosscorrelation")
        corr = calculate_corr(X)
        print_correlation_table(corr, min_corr=0.7)
        views.correlation_matrix(prj, corr, args.img_only)

    if args.pca:
        log.info("computing pca")
        pca = calculate_pca(X)
        log.info("computing pca projection")
        views.pca_projection(prj, pca, X, y, False)
        if args.D3:
            views.pca_projection(prj, pca, X, y, args.D3)
        views.pca_explained_variance(prj, pca, args.img_only)

    if args.stats:
        log.info("computing features stats")
        print_stats_table(X)

    inertia = False
    if args.cluster:
        if args.cluster_alg == 'kmeans':
            cluster_alg = kmeans_clustering
            if not args.nclusters:
                args.nclusters = len(set(np.argmax(y, axis=1)))
            args.nclusters = int(args.nclusters)
            if args.nmaxclusters:
                log.info("performing inertia analysis with clusters in the range (%d, %d)" % (args.nclusters, args.nmaxclusters))
                inertia = True
                n_clusters_analysis(X, args.nmaxclusters, args.nclusters)
            else:
                log.info("computing kmeans clustering with k=%d" % args.nclusters)
        elif args.cluster_alg == 'dbscan':
            cluster_alg = dbscan_clustering
            if not args.nclusters:
                args.nclusters = 2
            log.info("computing dbscan clustering with eps=%f" % args.nclusters)
            if args.nmaxclusters:
                log.warning("nmax specified but not used. Inertia analysis only available for Kmeans.")
        if not args.pca and not inertia:
            log.info("computing pca to plot clusters")
            pca = calculate_pca(X)
        if not inertia:
            ca = cluster_alg(X, args.nclusters)
            if len(set(ca.labels_)) == 1:
                log.error("clustering failed. Check input parameter.")
                quit()
            views.plot_clusters(prj, pca, X, y, ca, False)
            if args.D3:
                views.plot_clusters(prj, pca, X, y, ca, args.D3)

    views.show(args.img_only)
