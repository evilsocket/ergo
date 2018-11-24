import os
import argparse
import logging as log

from terminaltables import AsciiTable

from ergo.project import Project

def usage():
    print("usage: ergo cmp <path_1> <path_2> --dataset <path>")
    quit()

def parse_args(argv):
    parser = argparse.ArgumentParser(description="comparer")
    parser.add_argument("-d", "--dataset", dest = "dataset", action = "store", type = str, required = True)
    args = parser.parse_args(argv)
    return args

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

        table = [ ["Name", "Ref", "New", "Delta"] ]
        for label, ref_run in ref_repo.items():
            for name, ref_value in ref_run.items():
                new_value = new_repo[label][name]
                if new_value != ref_value:
                    delta = new_value - ref_value
                    sign  = '+' if delta >= 0 else ''
                    table.append( [\
                        "%s / %s" % (label, name),
                        "%.2f" % ref_value,
                        "%.2f" % new_value,
                        "%s%.2f" % (sign, delta)] )
        print("") 
        print("Report:")
        print(AsciiTable(table).table)
        
        heads = [""]
        for i in range(0, ref_cm.shape[0]):
            heads.append("class %d" % i)

        table = [ heads ]
        for i in range(0, ref_cm.shape[0]):
            row = ["class %d" % i]
            for j in range(0, ref_cm.shape[1]):
                ref_v = ref_cm[i][j]
                new_v = new_cm[i][j]
                
                if ref_v != new_v:
                    delta = new_v - ref_v
                    sign  = '+' if delta >= 0 else ''
                    cell = "%d (%s%d)" % (new_v, sign, delta)
                else:
                    cell = "%d" % ref_v

                row.append(cell)

            table.append(row)
 
        print("")
        print("Confusion matrix:")
        print(AsciiTable(table).table)
