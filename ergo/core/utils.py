import os
import shutil
import logging as log

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
