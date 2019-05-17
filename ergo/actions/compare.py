import os
import json
import argparse
import logging as log
import numpy as np
import tempfile

from terminaltables import AsciiTable

from ergo.project import Project

def parse_args(argv):
    parser = argparse.ArgumentParser(prog="ergo cmp", description="Compare the performances of two models against a given dataset.")

    parser.add_argument("path_1", help="Path of the first project to compare.")
    parser.add_argument("path_2", help="Path of the second project to compare.")

    parser.add_argument("-d", "--dataset", dest="dataset", action="store", type=str, required=True,
        help="The dataset file to use to compare the models.")
    parser.add_argument("-j", "--to-json", dest="to_json", action="store", type=str, required=False,
        help="Output the comparision results to json.")

    args = parser.parse_args(argv)
    return args

def red(s):
    return "\033[31m" + s + "\033[0m"

def green(s):
    return "\033[32m" + s + "\033[0m"

# https://stackoverflow.com/questions/11942364/typeerror-integer-is-not-json-serializable-when-serializing-json-in-python
def default(o):
    if isinstance(o, np.int64): return int(o)
    raise TypeError

def generate_reduced_dataset(dataset, size = 10):
    temp_name = next(tempfile._get_candidate_names())
    tmpfile = open(temp_name, 'w')
    count = 0
    with open(dataset, 'r') as inpfile:
        for line in inpfile: count += 1

    size = min(count, 100)
    size = max(count // 100, size)
    log.info("creating temporal file %s of size %d", temp_name, size)

    with open(dataset,'r') as inpfile:
        p = float(size)/ count
        # assuming header:
        tmpfile.write(inpfile.readline())
        for i in inpfile:
            line = inpfile.readline()
            if np.random.random() < p:
                tmpfile.write(line)
    tmpfile.close()
    return temp_name

def are_preparation_equal(projects, dataset):
    inp_shape = None
    for prj in projects:
        if inp_shape is None:
            inp_shape = prj.model.input_shape
        elif inp_shape != prj.model.input_shape:
            return False
        prj.prepare(dataset, 0.0, 0.0, False)
    return compare_datasets(projects[0].dataset, projects[1].dataset)


def compare_datasets(ds1, ds2):
    if ds1.is_flat:
        return np.array_equal(ds1.X, ds2.X)
    for i,j in zip(ds1.X, ds2.X):
        if not np.array_equal(i, j):
            return False
    return True


def action_compare(argc, argv):
    args     = parse_args(argv)
    metrics  = {}
    projects = { \
        args.path_1: None,
        args.path_2: None,
    }
    ref       = None
    inp_shape = None
    out_shape = None
    is_prepared = None
    prjs = []

    for path in projects:
        prj = Project(path)
        err = prj.load()
        if err is not None:
            log.error("error while loading project %s: %s", path, err)
            quit()
        prjs.append(prj)
        if not is_prepared:
            is_prepared = True
        else:
            small_dataset = generate_reduced_dataset(args.dataset)
            are_equal = are_preparation_equal(prjs, small_dataset)
            log.info("deleting temporal file %s", small_dataset)
            os.remove(small_dataset)
        if out_shape is None:
            out_shape = prj.model.output_shape
        elif out_shape != prj.model.output_shape:
            log.error("model %s output shape is %s, expected %s", path, prj.model.output_shape, out_shape)
            quit()
        projects[path] = prj

    for prj,path in zip(prjs, projects):
        prj = Project(path)
        err = prj.load()

        if err is not None:
            log.error("error while loading project %s: %s", path, err)
            quit()

        if ref is None:
            prj.prepare(args.dataset, 0, 0, False)
            ref = prj
            is_prepared = True
        else:
            if are_equal:
                log.info("Projects use same prepare.py file ...")
                prj.dataset.X, prj.dataset.Y = ref.dataset.X.copy(), ref.dataset.Y.copy()
            else:
                log.info("Projects use different prepare.py files, reloading dataset ...")
                prj.prepare(args.dataset, 0., 0., False)

        # TODO: Run in parallel?
        log.debug("running %s ...", path)
        metrics[path] = prj.accuracy_for(prj.dataset.X, prj.dataset.Y, repo_as_dict = True)

    prev = None
    for path, m in metrics.items():
        if prev is None:
            prev = m
            continue

        ref_repo, ref_cm = prev
        new_repo, new_cm = m
        diffs = {
            'report':[],
            'cm':[],
            'cm_stats':{}
        }

        table = [ ["Name", "Ref", "New", "Delta"] ]
        for label, ref_run in ref_repo.items():
            for name, ref_value in ref_run.items():
                new_value = new_repo[label][name]
                if new_value != ref_value:
                    delta = new_value - ref_value
                    sign, fn  = ('+', green) if delta >= 0 else ('', red)
                    diffs['report'].append({
                        'name': '%s / %s' % (label, name),
                        'delta': delta,
                    })
                    table.append( [\
                        "%s / %s" % (label, name),
                        "%.2f" % ref_value,
                        "%.2f" % new_value,
                        fn("%s%.2f" % (sign, delta))] )
        print("")
        print(AsciiTable(table).table)

        heads = [""]
        for i in range(0, ref_cm.shape[0]):
            heads.append("class %d" % i)

        table = [ heads ]
        total = 0
        impr  = 0
        regr  = 0
        for i in range(0, ref_cm.shape[0]):
            row = ["class %d" % i]
            row_diffs = []
            for j in range(0, ref_cm.shape[1]):
                ref_v = ref_cm[i][j]
                new_v = new_cm[i][j]
                total = total + new_v
                delta = new_v - ref_v

                if ref_v != new_v:
                    sign  = '+' if delta >= 0 else ''

                    if i == j:
                        fn = green if delta >= 0 else red
                    else:
                        fn = red if delta >= 0 else green

                    if fn == green:
                        impr += abs(delta)
                    else:
                        regr += abs(delta)

                    cell = fn("%d (%s%d)" % (new_v, sign, delta))
                else:
                    cell = "%d" % ref_v

                row.append(cell)
                row_diffs.append(delta)

            diffs['cm'].append(row_diffs)
            table.append(row)

        print("")
        print(AsciiTable(table).table)

        diffs['cm_stats'] = {
            'improvements': {
                'total': impr,
                'perc': impr / float(total) * 100.0
            },
            'regressions': {
                'total': regr,
                'perc': regr / float(total) * 100.0
            }
        }

        print("")
        print("Improvements: %d ( %.2f %% )" % (impr, impr / float(total) * 100.0))
        print("Regressions : %d ( %.2f %% )" % (regr, regr / float(total) * 100.0))

        if args.to_json is not None:
            print("")
            log.info("creating %s ...", args.to_json)
            with open(args.to_json, 'w+') as fp:
                json.dump(diffs, fp, default=default)
