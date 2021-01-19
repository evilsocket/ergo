import argparse
import json
import sys
stderr = sys.stderr
sys.stderr = open('/dev/null', 'w')
import tensorflow.keras as keras
import sklearn
sys.stderr = stderr

import tensorflow as tf
from tensorflow.python.client import device_lib

from ergo.version import banner, __version__

def get_pads(devs):
    namepad = 0
    typepad = 0
    for dev in devs:
        lname = len(dev.name)
        ltype = len(dev.device_type)

        if lname > namepad:
            namepad = lname

        if ltype > typepad:
            typepad = ltype

    return namepad, typepad

# https://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size
def mem_fmt(num, suffix='B'):
    num = int(num)
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

def parse_args(argv):
    parser = argparse.ArgumentParser(prog="ergo info", description="Print library versions and hardware info.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-j", "--to-json", dest="to_json", action="store_true", default=False,
        help="Output the information to json instead of text.")

    args = parser.parse_args(argv)
    return args

def action_info(argc, argv):
    args = parse_args(argv)

    if args.to_json is True:
        devices = []
        info = {
            "version": __version__,
            "keras_version": keras.__version__,
            "tf_version": tf.__version__,
            'sklean_version': sklearn.__version__,
            "devices":[]
        }
        devs = device_lib.list_local_devices()
        for dev in devs:
            info['devices'].append({
                'name': dev.name,
                'type': dev.device_type,
                'memory': dev.memory_limit,
                'description': dev.physical_device_desc
            })

        print(json.dumps(info))
    else:
        print(banner.strip("\n") % (__version__, keras.__version__, tf.__version__, sklearn.__version__))
        print("")
        print("Hardware:\n")

        devs = device_lib.list_local_devices()
        npad, tpad  = get_pads(devs)
        for dev in devs:
            print( "%s (%s) - %s" % (dev.name.ljust(npad,' '), dev.device_type.ljust(tpad,' '), mem_fmt(dev.memory_limit)))
