from setuptools import setup, find_packages
from ergo.version import __version__, __author__, __email__, __license__

import os

try:
    long_description = open( 'README.md', 'rt' ).read()
    ctype = 'text/markdown'
except:
    long_description = 'ergo is a tool that makes deep learning with Keras easier.'
    ctype = 'text/plain'

setup( name                 = 'ergo',
       version              = __version__,
       description          = long_description,
       long_description     = long_description,
       long_description_content_type = ctype,
       author               = __author__,
       author_email         = __email__,
       url                  = 'http://www.github.com/evilsocket/ergo',
       packages             = find_packages(),
       scripts              = [ 'bin/ergo' ],
       license              = __license__,
       classifiers          = [
            'Programming Language :: Python :: 3',
            'Development Status :: 5 - Production/Stable',
            'License :: OSI Approved :: GNU General Public License (GPL)',
            'Environment :: Console',
      ],
)
