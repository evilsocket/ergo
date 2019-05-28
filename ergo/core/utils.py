import os
import shutil
import logging as log
import numpy as np

from collections import defaultdict

def clean_if_exist(path, files):
    path  = os.path.abspath(path)
    for filename in files:
        filename = os.path.join(path, filename)
        if os.path.exists(filename):
            if os.path.isdir(filename):
                log.info("removing folder %s", filename)
                shutil.rmtree(filename)
            else:
                log.info("removing file %s", filename)
                os.remove(filename)

def serialize_classification_report(cr):
    tmp = list()
    for row in cr.split("\n"):
        parsed_row = [x for x in row.split("  ") if len(x) > 0]
        if len(parsed_row) > 0:
            tmp.append(parsed_row)
    
    measures = tmp[0]
    out = defaultdict(dict)
    for row in tmp[1:]:
        columns = len(row)
        class_label = row[0].strip()
        for j, m in enumerate(measures):
            # fixes https://github.com/evilsocket/ergo/issues/5
            value = float(row[j + 1].strip()) if j + 1 < columns else None
            metric = m.strip()
            out[class_label][metric] = value

    return out

def serialize_cm(cm):
    return cm.tolist()
