import os
import time
import sys
import glob
import argparse
import logging as log
import multiprocessing

from ergo.project import Project
from ergo.core.queue import TaskQueue

def parse_args(argv):
    parser = argparse.ArgumentParser(prog="ergo encode", description="Encode one or more files to vectors and create or update a csv dataset for training.")

    parser.add_argument("project", help="The path containing the model definition.")
    parser.add_argument("path", help="Path of a single file or of a folder of files.")

    parser.add_argument( "-l", "--label", dest="label", default='auto',
        help="The class of the file(s) or 'auto' to use the name of the containing folder.")
    parser.add_argument( "-o", "--output", dest="output", default='dataset.csv',
        help="Output CSV file names.")
    parser.add_argument( "-f", "--filter", dest="filter", default='*.*',
        help="If PATH is a folder, this flag determines which files are going to be selected.")
    parser.add_argument( "-m", "--multi", dest="multi", default=False, action="store_true",
        help="If PATH is a single file, this flag will enable line by line reading of multiple inputs from it.")
    parser.add_argument( "-w", "--workers", dest="workers", default=0, type=int,
        help="Number of concurrent workers to use for encoding, or zero to run two workers per available CPU core.")
    parser.add_argument( "-d", "--delete", dest="delete", default=False, action="store_true",
        help="Delete each file after encoding it.")

    args = parser.parse_args(argv)
    return args

# given the arguments setup and a path, return the label
def label_of(args, path):
    if args.label == 'auto':
        return os.path.basename(os.path.dirname(path))
    else:
        return args.label

progress_at = None
prev_done = 0
prev_speed = 0

def get_speed(done):
    global progress_at, prev_done, prev_speed

    speed = 0
    now   = time.time()
    if progress_at is not None:
        delta_v = int(done - prev_done)
        if delta_v >= 10:
            delta_t = now - progress_at
            speed = delta_v / delta_t
        else:
            speed = prev_speed
            now   = progress_at
            done  = prev_done

    progress_at = now
    prev_done = done
    prev_speed = speed

    return speed

# simple progress bar
def on_progress(done, total):
    maxsz = 80
    perc  = done / total
    sp    = " " if perc > 0 else ""
    bar   = ("█" * int(maxsz * perc)) + sp

    sys.stdout.write("\r%d/%d (%d/s) %s%.1f%%" % (done, total, get_speed(done), bar, perc * 100.0))
    if perc == 1.0:
        sys.stdout.write("\n")

# wait for lines on the results queue and append them to filepath
def appender(filepath, total, results):
    log.debug("file appender for %s started ...", filepath)

    print()
    with open(filepath, 'a+t') as fp:
        done = 0
        while True: 
            on_progress(done, total)

            line = results.get()
            if line is None:
                break

            fp.write(line.strip() + "\n")
            done += 1

# use the project prepare_input function to encode a single
# input into a CSV line
def parse_input(prj, inp, label, results, do_delete = False):
    log.debug("encoding '%s' as '%s' ...", inp, label)
    x = prj.logic.prepare_input(inp, is_encoding = True)
    results.put("%s,%s" % (str(label), ','.join(map(str,x))))
    if do_delete and os.path.isfile(inp): 
        os.remove(inp)

def action_encode(argc, argv):
    args = parse_args(argv)
    if not os.path.exists(args.path):
        log.error("%s does not exist.", args.path)
        quit()

    prj = Project(args.project)
    err = prj.load()
    if err is not None:
        log.error("error while loading project: %s", err)
        quit()
    
    args.label = args.label.strip().lower()
    log.info("using %s labeling", 'auto' if args.label == 'auto' else 'hardcoded')

    inputs = []
    if os.path.isdir(args.path):
        in_files = []
        if args.label == 'auto':
            # the label is inferred from the dirname, so we expect
            # args.path to contain multiple subfolders
            for subfolder in glob.glob(os.path.join(args.path, "*")):
                log.info("enumerating %s ...", subfolder)
                in_filter = os.path.join(subfolder, args.filter)
                in_sub    = glob.glob(in_filter)
                n_sub     = len(in_sub)
                if n_sub > 0:
                    log.info("collected %d inputs from %s", n_sub, subfolder)
                    in_files.extend(in_sub)
        else:
            # grab files directly from args.path
            in_filter = os.path.join(args.path, args.filter)
            in_files.extend(glob.glob(in_filter))
            log.info("collected %d inputs from %s", len(in_files), args.path)

        log.info("labeling %d files ...", len(in_files))
        for filepath in in_files:
            if os.path.isfile(filepath):
                inputs.append((label_of(args, filepath), filepath))

    elif args.multi:
        log.info("parsing multiple inputs from %s ...", args.path)
        label = label_of(args, args.path)
        with open(args.path, 'rt') as fp:
            for line in fp:
                inputs.append((label, line))

    else:
        label = label_of(args, args.path)
        inputs.append((label, args.path))

    # one encoding queue that pushes to another queue that centralizes
    # append operations to a single writer process
    num_in = len(inputs)
    enc_q = TaskQueue('encoding', args.workers)
    res_q = multiprocessing.Queue()
    app_p = multiprocessing.Process(target=appender, args=(args.output, num_in, res_q))

    # open the output file and start waiting for lines to append
    app_p.start()

    log.info("encoding %d inputs to %s ...", num_in, args.output)
    for (y, x) in inputs:
        enc_q.add_task(parse_input, prj, x, y, res_q, args.delete)

    # wait for all inputs to be encoded
    enc_q.join()
    # let the writer know there are no more inputs to read
    res_q.put(None)
    # wait for the writer to finish
    app_p.join()

