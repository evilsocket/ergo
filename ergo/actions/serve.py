import os
import argparse
import traceback
import logging as log
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

from ergo.project import Project

prj = None
app = Flask(__name__)

@app.route('/')
def route():
    global prj

    xin = request.args.get('x')
    if xin is None:
        return "Missing 'x' parameter.", 400

    try:
        x = prj.logic.prepare_input(xin)
        y = prj.model.predict(np.array([x]))
        return jsonify(y.tolist())
    except Exception as e:
        #log.exception("error while predicting on %s", x)
        traceback.print_exc()
        log.error("%s", e)
        return str(e), 400

def parse_args(argv):
    parser = argparse.ArgumentParser(prog="ergo serve", description="Load a model and expose an API that can be used to run its inference.")

    parser.add_argument("path", help="Path of the project.")

    parser.add_argument("--host", dest="host", action="store", type=str,
        help="Address or hostname to bind to.")
    parser.add_argument("--port", dest="port", action="store", type=int, default=8080,
        help="TCP port to bind to.")
    parser.add_argument("--debug", dest="debug", action="store_true", default=False,
        help="Enable debug messages.")

    args = parser.parse_args(argv)
    return args

def action_serve(argc, argv):
    global prj, app

    args = parse_args(argv)
    prj = Project(args.path)
    err = prj.load()
    if err is not None:
        log.error("error while loading project: %s", err)
        quit()
    elif not prj.is_trained():
        log.error("no trained Keras model found for this project")
        quit()

    app.run(host=args.host, port=args.port, debug=args.debug)
