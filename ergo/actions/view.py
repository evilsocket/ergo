import os
import logging as log

from ergo.project import Project

def usage():
    print("usage: ergo view <path>")
    quit()

def action_view(argc, argv):
    if argc < 1:
        usage()

    prj = Project(argv[0])
    err = prj.load()
    if err is not None:
        log.error("error while loading project: %s", err)
        quit()

    prj.view()
