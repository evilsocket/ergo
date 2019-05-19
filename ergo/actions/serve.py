import os
import argparse
import traceback
import logging as log
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

from ergo.project import Project

prj         = None
classes     = None
num_outputs = 0
app         = Flask(__name__)

@app.route('/')
def route():
    global prj, classes, num_outputs

    try:
        xin = request.args.get('x')
        if xin is None:
            return "missing 'x' parameter", 400

        # encode the input
        x = prj.logic.prepare_input(xin)
        # run inference
        y = prj.model.predict(np.array([x]))[0].tolist()
        # decode results
        num_y = len(y)
        resp = {}

        if num_y != num_outputs:
            return "expected %d output classes, got inference with %d results" % (num_outputs, num_y), 500

        resp = { classes[i] : y[i] for i in range(num_y) }

        return jsonify(resp), 200
    except Exception as e:
        return str(e), 400

def parse_args(argv):
    parser = argparse.ArgumentParser(prog="ergo serve", description="Load a model and expose an API that can be used to run its inference.")

    parser.add_argument("path", help="Path of the project.")

    parser.add_argument("-a", "--address", dest="address", action="store", type=str,
        help="IP address or hostname to bind to.")
    parser.add_argument("-p", "--port", dest="port", action="store", type=int, default=8080,
        help="TCP port to bind to.")
    parser.add_argument("-d", "--debug", dest="debug", action="store_true", default=False,
        help="Enable debug messages.")

    parser.add_argument("--classes", dest="classes", default=None,
        help="Optional comma separated list of output classes.")

    args = parser.parse_args(argv)
    return args

def action_serve(argc, argv):
    global prj, app, classes, num_outputs

    args = parse_args(argv)
    prj = Project(args.path)
    err = prj.load()
    if err is not None:
        log.error("error while loading project: %s", err)
        quit()
    elif not prj.is_trained():
        log.error("no trained Keras model found for this project")
        quit()

    if args.classes is None:
        num_outputs = prj.model.output.shape[1]
        classes = ["class_%d" % i for i in range(num_outputs)]
    else:
        classes = [s.strip() for s in args.classes.split(',') if s.strip() != ""]
        num_outputs = len(classes)

    app.run(host=args.address, port=args.port, debug=args.debug)
