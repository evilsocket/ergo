from setuptools import setup, find_packages
from ergo.version import __version__, __author__, __email__, __license__

import os


desc = 'ergo is a tool that makes deep learning with Keras easier.'

required = []
with open('requirements.txt') as fp:
    for line in fp:
        line = line.strip()
        if line != "":
            required.append(line)

setup( name                 = 'ergo-ai',
       version              = __version__,
       description          = desc,
       long_description     = desc,
       long_description_content_type = 'text/plain',
       author               = __author__,
       author_email         = __email__,
       url                  = 'http://www.github.com/evilsocket/ergo',
       packages             = find_packages(),
       install_requires     = required,
       scripts              = [ 'bin/ergo' ],
       license              = __license__,
       classifiers          = [
            'Programming Language :: Python :: 3',
            'Development Status :: 5 - Production/Stable',
            'License :: OSI Approved :: GNU General Public License (GPL)',
            'Environment :: Console',
      ],
)
