"""Convert a Keras model to tensorflow protobuf format.
"""
import os
import argparse

import tensorflow as tf
from tensorflow.keras import backend as K

import logging as log

from ergo.project import Project

# @credits to https://stackoverflow.com/questions/45466020/how-to-export-keras-h5-to-tensorflow-pb
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    log.info("freezing session and creating pruned computation graph ...")

    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

def parse_args(argv):
    parser = argparse.ArgumentParser(prog="ergo to-tf", description="Convert the model inside an ergo project to TensorFlow format.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("path", help="Path of the project containing the model.")
    args = parser.parse_args(argv)
    return args

def action_to_tf(argc, argv):
    args = parse_args(argv)
    prj = Project(args.path)
    err = prj.load()
    if err is not None:
        log.error("error while loading project: %s", err)
        quit()
    elif not prj.is_trained():
        log.error("no trained Keras model found for this projec")
        quit()

    frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in prj.model.outputs])

    log.info("saving protobuf to %s ...", os.path.join(prj.path, 'model.pb'))

    tf.train.write_graph(frozen_graph, prj.path, "model.pb", as_text=False)
