import os
import json
import argparse
import logging as log
import numpy as np

from terminaltables import AsciiTable

from ergo.project import Project

def usage():
    print("usage: ergo cmp <path_1> <path_2> --dataset <path> (--to-json <file>)")
    quit()

def parse_args(argv):
    parser = argparse.ArgumentParser(description="comparer")
    parser.add_argument("-d", "--dataset", dest = "dataset", action = "store", type = str, required = True)
    parser.add_argument("-j", "--to-json", dest = "to_json", action = "store", type = str, required = False)
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

def action_compare(argc, argv):
    if argc < 4:
        usage()

    args     = parse_args(argv[2:])
    metrics  = {}
    projects = { \
        argv[0]: None,
        argv[1]: None,
    }
    ref       = None
    inp_shape = None
    out_shape = None

    for path in projects:
        prj = Project(path)
        err = prj.load()
        if err is not None:
            log.error("error while loading project %s: %s", path, err)
            quit()

        if inp_shape is None:
            inp_shape = prj.model.input_shape

        elif inp_shape != prj.model.input_shape:
            log.error("model %s input shape is %s, expected %s", path, prj.model.input_shape, inp_shape)
            quit()

        if out_shape is None:
            out_shape = prj.model.output_shape

        elif out_shape != prj.model.output_shape:
            log.error("model %s output shape is %s, expected %s", path, prj.model.output_shape, out_shape)
            quit()

        if ref is None:
            ref = prj

        projects[path] = prj
        metrics[path] = None

    ref.prepare(args.dataset, 0.0, 0.0)

    log.info("evaluating %d models on %d samples ...", len(projects), len(ref.dataset.X))

    for path, prj in projects.items():
        # TODO: Run in parallel?
        log.debug("running %s ...", path)
        metrics[path] = prj.accuracy_for(ref.dataset.X, ref.dataset.Y, repo_as_dict = True)

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
