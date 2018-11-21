import keras

import tensorflow as tf
from tensorflow.python.client import device_lib

from ergo.version import __version__

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

def action_info(argc, argv):
    print("ergo %s" % __version__)
    print("keras  %s" % keras.__version__) 
    print("tf     %s" % tf.__version__)
    print("")

    devs = device_lib.list_local_devices()
    npad, tpad  = get_pads(devs)
    for dev in devs:
        print( "%s (%s) - %s" % (dev.name.ljust(npad,' '), dev.device_type.ljust(tpad,' '), mem_fmt(dev.memory_limit)))
