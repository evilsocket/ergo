import os
import argparse
import logging as log

import pandas as pd
from flask import Flask, request, jsonify

from ergo.project import Project

prj = None
app = Flask(__name__)

@app.route('/')
def route():
    global prj

    x = request.args.get('x')
    if x is None:
        return "Missing 'x' parameter.", 400

    try:
        x = prj.logic.prepare_input(x)
        y = prj.model.predict(x)
        return jsonify(y.tolist())
    except Exception as e:
        #log.exception("error while predicting on %s", x)
        log.error("%s", e)
        return str(e), 400

def parse_args(argv):
    parser = argparse.ArgumentParser(description="server")
    parser.add_argument("--host", dest = "host", action = "store", type = str)
    parser.add_argument("--port", dest = "port", action = "store", type = int, default = 8080)
    parser.add_argument("--debug", dest = "debug", action = "store_true", default = False)
    args = parser.parse_args(argv)
    return args

def usage():
    print("usage: ergo serve <path>")
    quit()

def action_serve(argc, argv):
    global prj, app

    if argc < 1:
        usage()

    args = parse_args(argv[1:])
    prj = Project(argv[0])
    err = prj.load()
    if err is not None:
        log.error("error while loading project: %s", err)
        quit()
    elif not prj.is_trained():
        log.error("no trained Keras model found for this project")
        quit()

    app.run(host=args.host, port=args.port, debug=args.debug)
