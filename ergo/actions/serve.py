import os
import argparse
import traceback
import threading
import logging as log
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

from ergo.project import Project

reqs        = 0
reload_lock = threading.Lock()
prj         = None
classes     = None
num_outputs = 0
app         = Flask(__name__)

def get_input(req):
    try:
        # search in query parameters
        x = request.args.get('x')
        if x is not None and x != "":
            return x
    except:
        pass

    try:
        # search in form body
        x = request.form.get('x')
        if x is not None and x != "":
            return x
    except:
        pass


    try:
        # search as file upload
        x = request.files['x']
        if x is not None:
            return x
    except:
        pass

    try:
        # search for direct json body
        x = req.get_json(silent=True)
        if x is not None:
            return x
    except:
        pass

    return None

@app.route('/encode', methods=['POST', 'GET'])
def encode_route():
    global prj, classes, num_outputs

    try:
        xin = get_input(request)
        if xin is None:
            return "missing 'x' parameter", 400

        x = prj.logic.prepare_input(xin)

        return jsonify(list(x)), 200
    except Exception as e:
        return str(e), 400

@app.route('/', methods=['POST', 'GET'])
def infer_route():
    global reqs, reload_lock, prj, classes, num_outputs
    try:
        xin = get_input(request)
        if xin is None:
            return "missing 'x' parameter", 400

        with reload_lock:
            # encode the input
            x = prj.logic.prepare_input(xin)
            # run inference
            y = prj.model.predict(np.array(x))[0].tolist()
            # decode results
            num_y = len(y)
            resp = {}

            if num_y != num_outputs:
                return "expected %d output classes, got inference with %d results" % (num_outputs, num_y), 500

            resp = { classes[i] : y[i] for i in range(num_y) }
            reqs += 1
            if reqs >= 25:
                prj.reload_model()
                reqs = 0

        return jsonify(resp), 200
    except Exception as e:
        log.exception("error while running inference")
        return str(e), 400

def parse_args(argv):
    parser = argparse.ArgumentParser(prog="ergo serve", description="Load a model and expose an API that can be used to run its inference.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("path", help="Path of the project.")

    parser.add_argument("-a", "--address", dest="address", action="store", type=str,
        help="IP address or hostname to bind to.")
    parser.add_argument("-p", "--port", dest="port", action="store", type=int, default=8080,
        help="TCP port to bind to.")
    parser.add_argument("--debug", dest="debug", action="store_true", default=False,
        help="Enable debug messages.")

    parser.add_argument("--profile", dest="profile", action="store_true", default=False,
        help="Enable profiler.")
    parser.add_argument("--prof-max", dest="restrictions", type=int, default=30,
        help="Maximum number of calls to profile.")

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
        if prj.classes is None:
            classes = ["class_%d" % i for i in range(num_outputs)]
        else:
            classes = [prj.classes[i] for i in range(num_outputs)]
    else:
        classes = [s.strip() for s in args.classes.split(',') if s.strip() != ""]
        num_outputs = len(classes)

    if args.profile:
        from werkzeug.contrib.profiler import ProfilerMiddleware
        args.debug = True
        app.config['PROFILE'] = True
        app.wsgi_app = ProfilerMiddleware(app.wsgi_app, restrictions=[args.restrictions])

    app.run(host=args.address, port=args.port, debug=args.debug)
