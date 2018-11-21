import sys
import os
import logging as log

from ergo.project import Project

def usage():
    log.info("usage: ergo create <path>")
    quit()

def action_create(argc, argv):
    if argc != 1:
        usage()
    
    path = argv[0]

    if os.path.exists(path):
        log.error("path %s already exists" % path)
        quit()
    
    log.info("creating %s ..." % path) 
    os.makedirs(path, exist_ok=True)

    Project.create(path)




